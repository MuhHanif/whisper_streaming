#!/usr/bin/env python3
import sys
import numpy as np
import librosa
import io
import soundfile
from functools import lru_cache
import time
import argparse
from faster_whisper import WhisperModel

import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import time
import threading


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "audio_path",
        type=str,
        help="Filename of 16kHz mono channel wav, on which live streaming is simulated.",
    )
    parser.add_argument(
        "--min-chunk-size",
        type=float,
        default=1.0,
        help="Minimum audio chunk size in seconds. It waits up to this time to do processing. If the processing takes shorter time, it waits, otherwise it processes the whole segment that was received by this time.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="base.en",
        choices="tiny.en,tiny,base.en,base,small.en,small,medium.en,medium,large-v1,large-v2,large".split(
            ","
        ),
        help="Name size of the Whisper model to use (default: large-v2). The model is automatically downloaded from the model hub if not present in model cache dir.",
    )
    parser.add_argument(
        "--model_cache_dir",
        type=str,
        default=None,
        help="Overriding the default model cache dir where models downloaded from the hub are saved",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Dir where Whisper model.bin and other files are saved. This option overrides --model and --model_cache_dir parameter.",
    )
    parser.add_argument(
        "--lan",
        "--language",
        type=str,
        default="en",
        help="Language code for transcription, e.g. en,de,cs.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Transcribe or translate.",
    )
    parser.add_argument(
        "--start_at",
        type=float,
        default=0.0,
        help="Start processing audio at this time.",
    )
    parser.add_argument(
        "--comp_unaware",
        action="store_true",
        default=False,
        help="Computationally unaware simulation.",
    )
    parser.add_argument(
        "--vad",
        action="store_true",
        default=True,
        help="Use VAD = voice activity detection, with the default parameters.",
    )
    return parser.parse_args()


@lru_cache
def load_audio(fname):
    a, _ = librosa.load(fname, sr=16000)
    return a


def load_audio_chunk(fname, beg, end):
    audio = load_audio(fname)
    beg_s = int(beg * 16000)
    end_s = int(end * 16000)
    return audio[beg_s:end_s]


# Whisper backend


class ASRBase:
    # join transcribe words with this character (" " for whisper_timestamped, "" for faster-whisper because it emits the spaces when neeeded)
    sep = " "

    def __init__(self, lan, modelsize=None, cache_dir=None, model_dir=None):
        self.transcribe_kargs = {}
        self.original_language = lan

        self.model = self.load_model(modelsize, cache_dir, model_dir)

    def load_model(self, modelsize, cache_dir):
        raise NotImplemented("must be implemented in the child class")

    def transcribe(self, audio, init_prompt=""):
        raise NotImplemented("must be implemented in the child class")

    def use_vad(self):
        raise NotImplemented("must be implemented in the child class")


class FasterWhisperASR(ASRBase):
    """Uses faster-whisper library as the backend. Works much faster, appx 4-times (in offline mode). For GPU, it requires installation with a specific CUDNN version.

    Requires imports, if used:
        import faster_whisper
    """

    sep = ""

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        from faster_whisper import WhisperModel

        if model_dir is not None:
            print(
                f"Loading whisper model from model_dir {model_dir}. modelsize and cache_dir parameters are not used.",
                file=sys.stderr,
            )
            model_size_or_path = model_dir
        elif modelsize is not None:
            model_size_or_path = modelsize
        else:
            raise ValueError("modelsize or model_dir parameter must be set")

        # this worked fast and reliably on NVIDIA L40
        model = WhisperModel(
            model_size_or_path,
            device="cuda",
            compute_type="float16",
            download_root=cache_dir,
        )
        return model

    def transcribe(self, audio, init_prompt=""):
        # tested: beam_size=5 is faster and better than 1 (on one 200 second document from En ESIC, min chunk 0.01)
        segments, info = self.model.transcribe(
            audio,
            language=self.original_language,
            initial_prompt=init_prompt,
            beam_size=5,
            word_timestamps=True,
            condition_on_previous_text=True,
            **self.transcribe_kargs,
        )
        return list(segments)

    def ts_words(self, segments):
        o = []
        for segment in segments:
            for word in segment.words:
                # not stripping the spaces -- should not be merged with them!
                w = word.word
                t = (word.start, word.end, w)
                o.append(t)
        return o

    def segments_end_ts(self, res):
        return [s.end for s in res]

    def use_vad(self):
        self.transcribe_kargs["vad_filter"] = True

    def set_translate_task(self):
        self.transcribe_kargs["task"] = "translate"


class HypothesisBuffer:
    def __init__(self):
        self.commited_in_buffer = []
        self.buffer = []
        self.new = []

        self.last_commited_time = 0
        self.last_commited_word = None

    def insert(self, new, offset):
        # compare self.commited_in_buffer and new. It inserts only the words in new that extend the commited_in_buffer, it means they are roughly behind last_commited_time and new in content
        # the new tail is added to self.new

        new = [(a + offset, b + offset, t) for a, b, t in new]
        self.new = [(a, b, t) for a, b, t in new if a > self.last_commited_time - 0.1]

        if len(self.new) >= 1:
            a, b, t = self.new[0]
            if abs(a - self.last_commited_time) < 1:
                if self.commited_in_buffer:
                    # it's going to search for 1, 2, ..., 5 consecutive words (n-grams) that are identical in commited and new. If they are, they're dropped.
                    cn = len(self.commited_in_buffer)
                    nn = len(self.new)
                    for i in range(1, min(min(cn, nn), 5) + 1):  # 5 is the maximum
                        c = " ".join(
                            [self.commited_in_buffer[-j][2] for j in range(1, i + 1)][
                                ::-1
                            ]
                        )
                        tail = " ".join(self.new[j - 1][2] for j in range(1, i + 1))
                        if c == tail:
                            print("removing last", i, "words:", file=sys.stderr)
                            for j in range(i):
                                print("\t", self.new.pop(0), file=sys.stderr)
                            break

    def flush(self):
        # returns commited chunk = the longest common prefix of 2 last inserts.

        commit = []
        while self.new:
            na, nb, nt = self.new[0]

            if len(self.buffer) == 0:
                break

            if nt == self.buffer[0][2]:
                commit.append((na, nb, nt))
                self.last_commited_word = nt
                self.last_commited_time = nb
                self.buffer.pop(0)
                self.new.pop(0)
            else:
                break
        self.buffer = self.new
        self.new = []
        self.commited_in_buffer.extend(commit)
        return commit

    def pop_commited(self, time):
        while self.commited_in_buffer and self.commited_in_buffer[0][1] <= time:
            self.commited_in_buffer.pop(0)

    def complete(self):
        return self.buffer


class OnlineASRProcessor:
    SAMPLING_RATE = 16000

    def __init__(self, asr, tokenizer):
        """asr: WhisperASR object
        tokenizer: sentence tokenizer object for the target language. Must have a method *split* that behaves like the one of MosesTokenizer.
        """
        self.asr = asr
        self.tokenizer = tokenizer

        self.init()

    def init(self):
        """run this when starting or restarting processing"""
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_time_offset = 0

        self.transcript_buffer = HypothesisBuffer()
        self.commited = []
        self.last_chunked_at = 0

        self.silence_iters = 0

    def insert_audio_chunk(self, audio):
        self.audio_buffer = np.append(self.audio_buffer, audio)

    def prompt(self):
        """Returns a tuple: (prompt, context), where "prompt" is a 200-character suffix of commited text that is inside of the scrolled away part of audio buffer.
        "context" is the commited text that is inside the audio buffer. It is transcribed again and skipped. It is returned only for debugging and logging reasons.
        """
        k = max(0, len(self.commited) - 1)
        while k > 0 and self.commited[k - 1][1] > self.last_chunked_at:
            k -= 1

        p = self.commited[:k]
        p = [t for _, _, t in p]
        prompt = []
        l = 0
        while p and l < 200:  # 200 characters prompt size
            x = p.pop(-1)
            l += len(x) + 1
            prompt.append(x)
        non_prompt = self.commited[k:]
        return self.asr.sep.join(prompt[::-1]), self.asr.sep.join(
            t for _, _, t in non_prompt
        )

    def process_iter(self):
        """Runs on the current audio buffer.
        Returns: a tuple (beg_timestamp, end_timestamp, "text"), or (None, None, "").
        The non-emty text is confirmed (commited) partial transcript.
        """

        prompt, non_prompt = self.prompt()
        print("PROMPT:", prompt, file=sys.stderr)
        print("CONTEXT:", non_prompt, file=sys.stderr)
        print(
            f"transcribing {len(self.audio_buffer)/self.SAMPLING_RATE:2.2f} seconds from {self.buffer_time_offset:2.2f}",
            file=sys.stderr,
        )
        res = self.asr.transcribe(self.audio_buffer, init_prompt=prompt)

        # transform to [(beg,end,"word1"), ...]
        tsw = self.asr.ts_words(res)

        self.transcript_buffer.insert(tsw, self.buffer_time_offset)
        o = self.transcript_buffer.flush()
        self.commited.extend(o)

        # there is a newly confirmed text
        if o:
            # we trim all the completed sentences from the audio buffer
            self.chunk_completed_sentence()

        # if the audio buffer is longer than 30s, trim it...
        if len(self.audio_buffer) / self.SAMPLING_RATE > 30:
            # ...on the last completed segment (labeled by Whisper)
            self.chunk_completed_segment(res)

            print(f"chunking because of len", file=sys.stderr)

        return self.to_flush(o)

    def chunk_completed_sentence(self):
        if self.commited == []:
            return
        print(self.commited, file=sys.stderr)
        sents = self.words_to_sentences(self.commited)
        for s in sents:
            print("\t\tSENT:", s, file=sys.stderr)
        if len(sents) < 2:
            return
        while len(sents) > 2:
            sents.pop(0)
        # we will continue with audio processing at this timestamp
        chunk_at = sents[-2][1]

        print(f"--- sentence chunked at {chunk_at:2.2f}", file=sys.stderr)
        self.chunk_at(chunk_at)

    def chunk_completed_segment(self, res):
        if self.commited == []:
            return

        ends = self.asr.segments_end_ts(res)

        t = self.commited[-1][1]

        if len(ends) > 1:
            e = ends[-2] + self.buffer_time_offset
            while len(ends) > 2 and e > t:
                ends.pop(-1)
                e = ends[-2] + self.buffer_time_offset
            if e <= t:
                print(f"--- segment chunked at {e:2.2f}", file=sys.stderr)
                self.chunk_at(e)
            else:
                print(f"--- last segment not within commited area", file=sys.stderr)
        else:
            print(f"--- not enough segments to chunk", file=sys.stderr)

    def chunk_at(self, time):
        """trims the hypothesis and audio buffer at "time" """
        self.transcript_buffer.pop_commited(time)
        cut_seconds = time - self.buffer_time_offset
        self.audio_buffer = self.audio_buffer[int(cut_seconds) * self.SAMPLING_RATE :]
        self.buffer_time_offset = time
        self.last_chunked_at = time

    def words_to_sentences(self, words):
        """Uses self.tokenizer for sentence segmentation of words.
        Returns: [(beg,end,"sentence 1"),...]
        """

        cwords = [w for w in words]
        t = " ".join(o[2] for o in cwords)
        s = self.tokenizer.split(t)
        out = []
        while s:
            beg = None
            end = None
            sent = s.pop(0).strip()
            fsent = sent
            while cwords:
                b, e, w = cwords.pop(0)
                if beg is None and sent.startswith(w):
                    beg = b
                elif end is None and sent == w:
                    end = e
                    out.append((beg, end, fsent))
                    break
                sent = sent[len(w) :].strip()
        return out

    def finish(self):
        """Flush the incomplete text when the whole processing ends.
        Returns: the same format as self.process_iter()
        """
        o = self.transcript_buffer.complete()
        f = self.to_flush(o)
        print("last, noncommited:", f, file=sys.stderr)
        return f

    def to_flush(
        self,
        sents,
        sep=None,
        offset=0,
    ):
        # concatenates the timestamped words or sentences into one sequence that is flushed in one line
        # sents: [(beg1, end1, "sentence1"), ...] or [] if empty
        # return: (beg1,end-of-last-sentence,"concatenation of sentences") or (None, None, "") if empty
        if sep is None:
            sep = self.asr.sep
        t = sep.join(s[2] for s in sents)
        if len(sents) == 0:
            b = None
            e = None
        else:
            b = offset + sents[0][0]
            e = offset + sents[-1][1]
        return (b, e, t)


WHISPER_LANG_CODES = "af,am,ar,as,az,ba,be,bg,bn,bo,br,bs,ca,cs,cy,da,de,el,en,es,et,eu,fa,fi,fo,fr,gl,gu,ha,haw,he,hi,hr,ht,hu,hy,id,is,it,ja,jw,ka,kk,km,kn,ko,la,lb,ln,lo,lt,lv,mg,mi,mk,ml,mn,mr,ms,mt,my,ne,nl,nn,no,oc,pa,pl,ps,pt,ro,ru,sa,sd,si,sk,sl,sn,so,sq,sr,su,sv,sw,ta,te,tg,th,tk,tl,tr,tt,uk,ur,uz,vi,yi,yo,zh".split(
    ","
)


def create_tokenizer(lan):
    """returns an object that has split function that works like the one of MosesTokenizer"""

    assert (
        lan in WHISPER_LANG_CODES
    ), "language must be Whisper's supported lang code: " + " ".join(WHISPER_LANG_CODES)

    if lan == "uk":
        import tokenize_uk

        class UkrainianTokenizer:
            def split(self, text):
                return tokenize_uk.tokenize_sents(text)

        return UkrainianTokenizer()

    # supported by fast-mosestokenizer
    if (
        lan
        in "as bn ca cs de el en es et fi fr ga gu hi hu is it kn lt lv ml mni mr nl or pa pl pt ro ru sk sl sv ta te yue zh".split()
    ):
        from mosestokenizer import MosesTokenizer

        return MosesTokenizer(lan)

    # the following languages are in Whisper, but not in wtpsplit:
    if (
        lan
        in "as ba bo br bs fo haw hr ht jw lb ln lo mi nn oc sa sd sn so su sw tk tl tt".split()
    ):
        print(
            f"{lan} code is not supported by wtpsplit. Going to use None lang_code option.",
            file=sys.stderr,
        )
        lan = None

    from wtpsplit import WtP

    # downloads the model from huggingface on the first use
    wtp = WtP("wtp-canine-s-12l-no-adapters")

    class WtPtok:
        def split(self, sent):
            return wtp.split(sent, lang_code=lan)

    return WtPtok()


##### FAST API PART! #####

app = FastAPI()
# Set up a global variable to keep track of the WebSocket connections
connected_websockets = set()

# Store the received audio data
audio_data = bytearray()

# TODO: hard code value for now (properly replace this!)
size = "base.en"
language = "en"
chunk = 1
SAMPLING_RATE = 16000

# load whisper
whisper_asr = FasterWhisperASR(modelsize=size, lan=language)
whisper_asr.use_vad()


async def send_transcription(websocket, preprocessor_obj, bytearray_data):
    if bytearray_data:
        # this is the function that sending the transcription back to websocket

        # convert to numpy and delete it
        chunk_audio = np.frombuffer(bytearray_data, dtype=np.int16)
        # convert it to float32 because insert_audio_chunk need fp32
        chunk_audio = chunk_audio.astype(np.float32)

        await websocket.send_text("test")
        preprocessor_obj.insert_audio_chunk(chunk_audio)


# simple demo url
@app.get("/")
async def serve_html():
    # Serve the HTML file
    with open("simple_web.html", "r") as html_file:
        content = html_file.read()
    return HTMLResponse(content)


# Define a function to send "testing" at a 5-second interval to all connected websockets
async def send_testing(audio_data, websocket, preprocessor_obj):
    # while True:
    # for websocket in connected_websockets:
    if audio_data:
        try:
            # convert audio to numpy so it can be processed by whisper
            audio, sample_rate = soundfile.read(
                io.BytesIO(audio_data),
                channels=1,
                samplerate=44100,
                dtype="float32",
                format="RAW",
                subtype="FLOAT",
                endian="LITTLE",
            )
            # debug save to wav
            # soundfile.write("testing.wav", audio, sample_rate, subtype='FLOAT', format='WAV', endian='LITTLE')
            target_sample_rate = 16000
            resampled_audio = librosa.resample(
                audio, orig_sr=sample_rate, target_sr=target_sample_rate
            )

            preprocessor_obj.insert_audio_chunk(resampled_audio)
            o = preprocessor_obj.process_iter()
            if o[0]:
                await websocket.send_text(
                    f"audio sample rate: {sample_rate} ==> transcription: {o}"
                )

        except Exception as e:
            await websocket.send_text(f"error {e}")


@app.websocket("/ws")
async def audio_stream(websocket: WebSocket):
    audio_data = bytearray()
    await websocket.accept()
    # init transcription concatenation logic
    online = OnlineASRProcessor(whisper_asr, create_tokenizer(language))
    beg = 0
    start = time.time() - beg
    end = 0
    min_chunk = 2
    initial_prompt = ""
    try:
        while True:
            # retrieve audio stream from websocket
            data = await websocket.receive_bytes()
            # stash audio data
            audio_data.extend(data)
            # keep stashing the bytes until the chunk time is filled
            now = time.time() - start
            if now < end + min_chunk:
                continue
            end = time.time() - start

            # simulate task
            np_start = time.time()
            asyncio.create_task(send_testing(audio_data, websocket, online))
            audio_data = bytearray()
            np_stop = time.time()

    except Exception as e:
        print(e)


@app.websocket("/ws_transcribe")
async def audio_stream(websocket: WebSocket):
    # store audio byte stream here
    audio_data = bytearray()

    # accept websocket connection
    await websocket.accept(bytearray)

    # init transcription concatenation logic
    online = OnlineASRProcessor(whisper_model, create_tokenizer(language))

    min_chunk = 5
    beg = 0
    start = time.time() - beg
    end = 0

    try:
        while True:
            # retrieve audio stream from websocket
            data = await websocket.receive_bytes()
            # stash audio data
            audio_data.extend(data)

            # keep stashing the bytes until the chunk time is filled
            now = time.time() - start
            if now < end + min_chunk:
                continue
            end = time.time() - start
            # send to transcribe process
            asyncio.create_task(
                send_transcription(
                    websocket=websocket,
                    preprocessor_obj=online,
                    bytearray_data=audio_data,
                )
            )
            # flush byte array
            audio_data = bytearray()
    except Exception as e:
        print(e)
