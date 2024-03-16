import preamble.dataset_maker as dm

path_originals_links = input("Enter path of links for original songs (see README.md):") 
path_versions_links  = input("Enter path of links for versions songs (see README.md):")

data_originals = dm.read_csv_and_transform(path_originals_links)
# print(data_originals)
data_versions  = dm.read_csv_and_transform(path_versions_links)
# print(data_versions)

for audio in data_originals:
    dm.download_audio_youtube(audio, "originals")

for audio in data_versions:
    dm.download_audio_youtube(audio, "versions")

