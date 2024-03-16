import os
import pytube
from pydub import AudioSegment
import csv

def read_csv_and_transform(csv_file):
    data = []
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        # Skip the header row
        # next(csv_reader)
        for row in csv_reader:
            # Extract name, version, and link from each row
            link = row[0]
            song_name = row[1]
            genre = row[2]
            # Append [name, version, link] to the data list
            data.append([link, song_name, genre])
    return data

# ### SPECIFY OUTPUT PATHS FOR COVERS AND ORIGINALS
# ORIGINALS_OUTPUT_PATH = 'data/originals'
# COVERS_OUTPUT_PATH = 'data/covers'

# ORIGINAL_LINKS = [
#     ### INSERT LINKS FOR WANTED ORIGINAL SONGS
#     ### MUST BE IN FORMAT (link, song_name, genre)
#     ### song_name should be in snake_case, genre can be left empty for originals
# ]

# COVER_LINKS = [
#     ### INSERT LINKS FOR WANTED COVER VERSIONS OF ORIGINAL SONGS
#     ### MUST BE IN FORMAT (link, song_name, genre) -- genre needed to ensure cover files dont overwrite each other
#     ### song_name should be in snake_case
# ]

def download_audio_youtube(audio, output_path):
    link, name, genre = audio
    # Download YouTube video
    youtube = pytube.YouTube(link)
    video_stream = youtube.streams.filter(only_audio=True).first()

    # Download video and convert to WAV
    audio_stream = video_stream.download(output_path="temp")
    audio = AudioSegment.from_file(audio_stream)

    # Ensure the output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    wav_filename = os.path.join(output_path, f"{name}_{genre}.wav")
    # Create the file before exporting audio
    with open(wav_filename, 'wb') as file:
        # Export audio to the file
        audio.export(file, format="wav")

        # print(f"Audio downloaded and saved as {wav_filename}")

    # except Exception as e:
    #     print("Error:", e)

# if __name__ == "__main__":
    # for link in ORIGINAL_LINKS:
    #     download_audio_youtube(link, ORIGINALS_OUTPUT_PATH, 'original')

    # for link in COVER_LINKS:
    #     download_audio_youtube(link, COVERS_OUTPUT_PATH, 'cover')
