# abc_to_wav.py (run in the same folder)
from music21 import converter
from midi2audio import FluidSynth

SOUNDFONT = r"C:\SoundFonts\FluidR3_GM.sf2"  # download a GM .sf2 once
abc = open("generated_abc.txt", encoding="utf-8").read()

s = converter.parseData(abc, format="abc")
s.write("midi", fp="out.mid")
FluidSynth(SOUNDFONT).midi_to_audio("out.mid", "out.wav")
print("Saved out.wav")
