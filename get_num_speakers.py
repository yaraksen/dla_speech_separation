from glob import glob

files = glob("data/datasets/speech_separation/train-clean-100/*-mixed.wav")
speakers = set()
for file in files:
    speakers.add(file.split("/")[-1].split("_")[0])

print(len(speakers))
