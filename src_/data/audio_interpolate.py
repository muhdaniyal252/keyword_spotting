import os
import librosa
import soundfile as sf


# Script expects two folder structures at the root level: "Export" & "Input" -> each with sub-folders per WakeWord
# Input-Folder as Input, interpolate all sounds in different ways, export new samples to Export-Folder
# LIBROSA effects doc: https://librosa.org/doc/main/effects.html
def interpolate_audio():

    # Inputs
    ha_input_path = "Input/Unknown"
    hh_input_path = "Input/Hilfe Hilfe"
    a_input_path = "Input/Adele"
    inputs = [ha_input_path, hh_input_path, a_input_path]

    for dirs in inputs:
        export_dir = "Export" + dirs[5:] + "/"

        for file in os.listdir(dirs):
            if file[0] == ".":
                continue

            # sr=None keeps original sampleRate
            audio, sr = librosa.load(dirs + "/" + file, sr=None)

            # faster = librosa.effects.time_stretch(audio, rate=1.05)
            # sf.write(export_dir + file[:-4] + "_*105.wav", faster, int(sr))
            # continue

            # Pitch-Shifting (-8, -6, -4, -2, +2, +4, +6, +8)
            shift_up_2 = librosa.effects.pitch_shift(audio, sr=sr, n_steps=2)
            shift_up_4 = librosa.effects.pitch_shift(audio, sr=sr, n_steps=4)
            shift_up_6 = librosa.effects.pitch_shift(audio, sr=sr, n_steps=6)
            shift_up_8 = librosa.effects.pitch_shift(audio, sr=sr, n_steps=8)
            shift_down_2 = librosa.effects.pitch_shift(audio, sr=sr, n_steps=-2)
            shift_down_4 = librosa.effects.pitch_shift(audio, sr=sr, n_steps=-4)
            shift_down_6 = librosa.effects.pitch_shift(audio, sr=sr, n_steps=-6)
            shift_down_8 = librosa.effects.pitch_shift(audio, sr=sr, n_steps=-8)
            sf.write(export_dir + file[:-4] + "_+2.wav", shift_up_2, int(sr))
            sf.write(export_dir + file[:-4] + "_+4.wav", shift_up_4, int(sr))
            sf.write(export_dir + file[:-4] + "_+6.wav", shift_up_6, int(sr))
            # sf.write(export_dir + file[:-4] + "_+8.wav", shift_up_8, int(sr))
            sf.write(export_dir + file[:-4] + "_-2.wav", shift_down_2, int(sr))
            sf.write(export_dir + file[:-4] + "_-4.wav", shift_down_4, int(sr))
            sf.write(export_dir + file[:-4] + "_-6.wav", shift_down_6, int(sr))
            # sf.write(export_dir + file[:-4] + "_-8.wav", shift_down_8, int(sr))

            # Time Stretch (*2, *1.5, *0.75, *0.5)
            double_time = librosa.effects.time_stretch(audio, rate=2.0)
            faster = librosa.effects.time_stretch(audio, rate=1.5)
            slower = librosa.effects.time_stretch(audio, rate=0.75)
            half_time = librosa.effects.time_stretch(audio, rate=0.5)
            sf.write(export_dir + file[:-4] + "_*200.wav", double_time, int(sr))
            sf.write(export_dir + file[:-4] + "_*150.wav", faster, int(sr))
            sf.write(export_dir + file[:-4] + "_*075.wav", slower, int(sr))
            sf.write(export_dir + file[:-4] + "_*050.wav", half_time, int(sr))

            # Emphasis (kinda like distortion)
            neg_emphasis = librosa.effects.preemphasis(audio, coef=0.8)
            emphasis = librosa.effects.deemphasis(audio, coef=0.8)
            sf.write(export_dir + file[:-4] + "_neg-emphasis.wav", neg_emphasis, int(sr))
            sf.write(export_dir + file[:-4] + "_emphasis.wav", emphasis, int(sr))


interpolate_audio()