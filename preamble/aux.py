import os
import sys
import numpy as np
import librosa
import matplotlib.pyplot as plt
from pydub import AudioSegment
from IPython.display import display, Audio
import essentia.standard as es
from essentia.pytools.spectral import hpcpgram
# import itertools

sys.path.append('..')
import libfmp.b
import libfmp.c4
import libfmp.c7

# %matplotlib inline

def load_audio(path, Fs):
    y_og, fs_og, n_channels, _, _, _ = es.AudioLoader(filename=path)()
    y_mono       = es.MonoMixer()(y_og, 2)
    y_mono_fs = es.Resample(inputSampleRate=fs_og, outputSampleRate=Fs, quality=1).compute(y_mono)
    return y_mono_fs

def change_tempo(input_folder, output_folder, tempo_change_factor, label):
  if not os.path.exists(output_folder):
    os.makedirs(output_folder)

  for filename in os.listdir(input_folder):
    input_file_path = os.path.join(input_folder, filename)
    y, sr = librosa.load(input_file_path)
    y_stretched = librosa.effects.time_stretch(y, rate=tempo_change_factor)

    output_file_path = os.path.join(output_folder, label + '_' + filename)
    sf.write(output_file_path, y_stretched, sr)

    print(f"Tempo changed for {filename} and saved as {output_file_path}")


def change_pitch(input_file_path, output_file_path, pitch_steps, label):
  if not os.path.exists(output_folder):
    os.makedirs(output_folder)

  for filename in os.listdir(input_folder):
    input_file_path = os.path.join(input_folder, filename)
    y, sr = librosa.load(input_file_path, sr=None)
    y_pitch_shifted = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=pitch_steps)

    output_file_path = os.path.join(output_folder, label + '_' + filename)
    sf.write(output_file_path, y_pitch_shifted, sr)

    print(f"Pitch changed for {filename} and saved as {output_file_path}")

def compute_sm_libfmp(path_1, path_2, Fs=22050, N=4410, H=2205, ell=21, d=5, L_smooth=12,
               tempo_rel_set=np.array([0.66, 0.81, 1, 1.22, 1.5]),
               shift_set=np.array([0]), strategy='relative', scale=True,
               thresh=0.15, penalty=-2.0, binarize=False, plotting=False):
    """Compute similarity score between two songs

    Args:
        path_1 (string): First signal path (any sound extension)
        path_2 (string): Second signal path (any sound extension)
        Fs (scalar): Sampling rate of WAV files
        N (int): Window size for computing STFT-based chroma features (Default value = 4410)
        H (int): Hop size for computing STFT-based chroma features (Default value = 2205)
        ell (int): Smoothing length for computing CENS features (Default value = 21)
        d (int): Downsampling factor for computing CENS features (Default value = 5)
        L_smooth (int): Length of filter for enhancing SM (Default value = 12)
        tempo_rel_set (np.ndarray): Set of relative tempo values for enhancing SM
            (Default value = np.array([0.66, 0.81, 1, 1.22, 1.5]))
        shift_set (np.ndarray): Set of shift indices for enhancing SM (Default value = np.array([0]))
        strategy (str): Thresholding strategy for thresholding SM ('absolute', 'relative', 'local')
            (Default value = 'relative')
        scale (bool): If scale=True, then scaling of positive values to range [0,1] for thresholding SM
            (Default value = True)
        thresh (float): Treshold (meaning depends on strategy) (Default value = 0.15)
        penalty (float): Set values below treshold to value specified (Default value = -2.0)
        binarize (bool): Binarizes final matrix (positive: 1; otherwise: 0) (Default value = False)

    Returns:
        X (np.ndarray): CENS feature sequence for first signal
        Y (np.ndarray): CENS feature sequence for second signal
        Fs_feature (scalar): Feature rate
        S_thresh (np.ndarray): Similarity matrix
        I (np.ndarray): Index matrix
    """
    # Print the paths being compared
    # print("Computing: ", path_1, " vs ", path_2)
    # Load audio files and extract chroma features
    x1 = load_audio(path_1, Fs)[Fs*30:-Fs*30]
    x2 = load_audio(path_2, Fs)[Fs*30:-Fs*30]
    C1 = librosa.feature.chroma_stft(y=x1, sr=Fs, tuning=0, norm=1, hop_length=H, n_fft=N)
    C2 = librosa.feature.chroma_stft(y=x2, sr=Fs, tuning=0, norm=1, hop_length=H, n_fft=N)
    Fs_C = Fs / H
    X, Fs_feature = libfmp.c7.compute_cens_from_chromagram(C1, Fs_C, ell=ell, d=d)
    Y, Fs_feature = libfmp.c7.compute_cens_from_chromagram(C2, Fs_C, ell=ell, d=d)
    # Compute the structural similarity matrix
    S, I = libfmp.c4.compute_sm_ti(X, Y, L=L_smooth, tempo_rel_set=tempo_rel_set,
                                   shift_set=shift_set, direction=2)
    # Threshold the similarity matrix
    S_thresh = libfmp.c4.threshold_matrix(S, thresh=thresh, strategy=strategy,
                                          scale=scale, penalty=penalty, binarize=binarize)
    # Compute the accumulated score matrix
    D = libfmp.c7.compute_accumulated_score_matrix_common_subsequence(S_thresh)
    Dmax = np.max(D)
    # Plot the results if plotting is enabled
    if plotting == True:
        cmap_penalty = libfmp.c4.colormap_penalty(penalty=penalty)
        n, m = divmod(np.argmax(D), D.shape[1])
        P = libfmp.c7.compute_optimal_path_common_subsequence(D)
        seg_X, seg_Y = libfmp.c7.get_induced_segments(P)
        Fs_X = Fs_feature
        # Create subplots for visualization
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        # Plot 1: Score matrix S
        axs[0, 0].imshow(S, cmap=cmap_penalty, aspect='equal')
        axs[0, 0].set_title('Score matrix $\mathbf{S}$')
        axs[0, 0].set_xlabel('Time (seconds)')
        axs[0, 0].set_ylabel('Time (seconds)')
        # Plot 2: Score matrix S with optimal path
        axs[0, 1].imshow(S, cmap=cmap_penalty, aspect='equal')
        axs[0, 1].plot(P[:, 1], P[:, 0], marker='.', color='r')
        axs[0, 1].set_title('Score matrix $\mathbf{S}$ with optimal path')
        axs[0, 1].set_xlabel('Time (frames)')
        axs[0, 1].set_ylabel('Time (frames)')
        # Plot 3: Accumulated score matrix D with optimal path
        axs[1, 0].imshow(D, cmap='gray_r', aspect='equal')
        axs[1, 0].plot(P[:, 1], P[:, 0], marker='.', color='r')
        axs[1, 0].set_title('Accumulated score matrix $\mathbf{D}$ with optimal path')
        axs[1, 0].set_xlabel('Time (frames)')
        axs[1, 0].set_ylabel('Time (frames)')
        # Plot 4: Score matrix S with optimal path and alignment
        axs[1, 1].imshow(S, cmap=cmap_penalty, aspect='equal')
        axs[1, 1].plot(P[:, 1] / Fs_X, P[:, 0] / Fs_X, marker='.', color='r')
        # Define the time segments
        start_X, start_Y = P[0, :] / Fs_X
        end_X, end_Y = P[-1, :] / Fs_X
        axs[1, 1].plot([0, 0], [start_X, end_X], c='g', linewidth=7)
        axs[1, 1].plot([start_Y, end_Y], [0, 0], c='g', linewidth=7)
        axs[1, 1].plot([0, start_Y], [start_X, start_X], c='r', linestyle=':')
        axs[1, 1].plot([0, end_Y], [end_X, end_X], c='r', linestyle=':')
        axs[1, 1].plot([start_Y, start_Y], [0, start_X], c='r', linestyle=':')
        axs[1, 1].plot([end_Y, end_Y], [0, end_X], c='r', linestyle=':')
        axs[1, 1].set_title('Score matrix $\mathbf{S}$ with optimal path and alignment')
        axs[1, 1].set_xlabel('Time (seconds)')
        axs[1, 1].set_ylabel('Time (seconds)')
        plt.tight_layout()
        plt.show()
        # Print the induced segments and play audio clips
        print('Induced segments:')
        display(Audio(x1[int(start_X * Fs):int(end_X * Fs)], rate=Fs))
        display(Audio(x2[int(start_Y * Fs):int(end_Y * Fs)], rate=Fs))
    return Dmax

def compute_sm_essentia(path_1, path_2, Fs=22050,
                                        frame_stack_size=9, frame_stack_stride=1, binarize_percentile=0.095, oti=True,
                                        alignment_type='serra09', dis_extension=0.5, dis_onset=0.5, distance_type='asymmetric'):
    """
    Generate a similarity matrix between two audio files using Essentia library.

    Args:
        song1_filename (str): Filename of the first audio file.
        song2_filename (str): Filename of the second audio file.
        frame_stack_size (int, optional): Size of the frame stack for chroma cross-similarity computation (default is 9).
        frame_stack_stride (int, optional): Stride of the frame stack for chroma cross-similarity computation (default is 1).
        binarize_percentile (float, optional): Percentile threshold for binarizing the cross-similarity matrix (default is 0.095).
        oti (bool, optional): Whether to use Oti (Offset-Time-Warping-Invariant) or not (default is True).
        alignment_type (str, optional): Alignment method for cover song similarity computation (default is 'serra09').
        dis_extension (float, optional): Extension parameter for cover song similarity computation (default is 0.5).
        dis_onset (float, optional): Onset parameter for cover song similarity computation (default is 0.5).
        distance_type (str, optional): Distance metric for cover song similarity computation (default is 'asymmetric').

    Returns:
        np.ndarray: Score matrix representing similarity between segments of the two audio files.
        float: Distance between the two audio files.
    """
    # print("Computing: ",path_1," vs ",path_2)

    x1 = load_audio(path_1, Fs)[Fs*30:-Fs*30]
    x2 = load_audio(path_2, Fs)[Fs*30:-Fs*30]

    song1_hpcpgram = hpcpgram(x1, sampleRate=44100)
    song2_hpcpgram = hpcpgram(x2, sampleRate=44100)

    cross_similarity = es.ChromaCrossSimilarity(frameStackSize=frame_stack_size,
                                        frameStackStride=frame_stack_stride,
                                        binarizePercentile=binarize_percentile,
                                        oti=oti)

    sim_matrix = cross_similarity(song1_hpcpgram, song2_hpcpgram)
    score_matrix, distance = es.CoverSongSimilarity(disOnset=dis_onset,
                                            disExtension=dis_extension,
                                            alignmentType=alignment_type,
                                            distanceType=distance_type)(sim_matrix)
    Dmax = np.max(score_matrix)
    return Dmax

def list_files_in_folder(path):
    file_paths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

def extract_name(path):
    base_filename = os.path.splitext(os.path.basename(path))[0]
    name = base_filename.replace('_original', '').replace('_original', '').replace('_original', '')
    name = name.replace("fast_","").replace("slow_","").replace("pitched_down_","").replace("pitched_up_","")
    return name

def tester(originals_path, covers_path):
    # # parameters
    # Fs = 22050
    # penalty = -2
    # tempo_rel_set = np.array([0.66, 0.81, 1, 1.22, 1.5])    
    # L_smooth = 12

    #actual function
    files_og     = list_files_in_folder(originals_path)
    files_covers = list_files_in_folder(covers_path)
    print(files_covers)
    results = {}
    for og in files_og:
        name = extract_name(og)
        results[name] = {}
        print("Now analyzing ", og, " with name ", name)
        for cover in files_covers:
            cover_name = base_filename = os.path.splitext(os.path.basename(cover))[0]
            if name in cover_name:
                print("   Match made between ", name, " and " ,cover_name)
                result_libfmp        = compute_sm_libfmp(og, cover)
                result_essentia      = compute_sm_essentia(og, cover)
                results[name][cover_name]   = [result_libfmp, result_essentia]
    return results

def dict_results_to_plot(dict_results, title):
    # Sort dictionary items alphabetically by keys
    sorted_list = sorted(dict_results.items())
    # Extract keys and values from the sorted list
    song_names = [item[0] for item in sorted_list]
    data_values = [item[1] for item in sorted_list]
    data1 = [data[0] for data in data_values]
    data2 = [data[1] for data in data_values]

    # Create subplots for two curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot curve 1 in the first subplot
    axes[0].plot(song_names, data1, label='libfmp', marker="x", linestyle = "")
    axes[0].set_xlabel('Song Names')
    axes[0].set_ylabel('Values')
    axes[0].set_title('libfmp')
    axes[0].axhline(y=100, color='r', linestyle='--', label='threshold')
    axes[0].tick_params(axis='x', rotation=90, labelsize=8)  # Rotate x-axis labels vertically
    axes[0].legend()
    axes[0].grid(True)

    # Plot curve 2 in the second subplot
    axes[1].plot(song_names, data2, label='essentia',  marker="x", linestyle = "")
    axes[1].set_xlabel('Song Names')
    axes[1].set_ylabel('Values')
    axes[1].set_title('essentia')
    axes[1].axhline(y=500, color='r', linestyle='--', label='threshold')
    axes[1].tick_params(axis='x', rotation=90, labelsize=8)  # Rotate x-axis labels vertically
    axes[1].legend()
    axes[1].grid(True)

    # Set the title for the entire plot
    plt.suptitle(title)

    # Adjust layout
    plt.tight_layout()

    # Show plot
    plt.show()

def display_by_name(name, path_versions):
    files = list_files_in_folder(path_versions)
    for file in files:
        name_temp = os.path.splitext(os.path.basename(file))[0]
        if name == name_temp:
            print(name)
            y = load_audio(file, 22050)[:22050*30]
            display(Audio(y, rate=22050))

def best_cases_display(results, path_versions):
    # Sort dictionary items alphabetically by keys
    sorted_list = sorted(results.items())
    # Extract keys and values from the sorted list
    song_names  = [item[0] for item in sorted_list]
    song_names  = song_names[1:]
    data_values = [item[1] for item in sorted_list]
    data1 = [data[0] for data in data_values]
    data2 = [data[1] for data in data_values]
    self_comp_1=data1[0]
    self_comp_2=data2[0]    
    data1 = data1[1:]
    data2 = data2[1:]
    # Find the indices of the best two scores in data0 and data1
    best_indices_data0 = sorted(range(len(data1)), key=lambda i: data1[i], reverse=True)[:2]
    best_indices_data1 = sorted(range(len(data2)), key=lambda i: data2[i], reverse=True)[:2]
    # Extract the names of the samples with the best scores in data0 and data1
    best_names_data0 = [song_names[i] for i in best_indices_data0]
    best_names_data1 = [song_names[i] for i in best_indices_data1]
    # Find the indices of the best two scores in data0 and data1
    worst_indices_data0 = sorted(range(len(data1)), key=lambda i: data1[i], reverse=False)[:2]
    worst_indices_data1 = sorted(range(len(data2)), key=lambda i: data2[i], reverse=False)[:2]
    # Extract the names of the samples with the best scores in data0 and data1
    worst_names_data0 = [song_names[i] for i in worst_indices_data0]
    worst_names_data1 = [song_names[i] for i in worst_indices_data1]

    print("\033[1mBest essentia:\033[0m")
    for best in best_names_data0:
        display_by_name(best,path_versions)
    print("\033[1mBest libfmp:\033[0m")
    for best in best_names_data1:
        display_by_name(best,path_versions)
    print("")
    print("\033[1mWorst essentia:\033[0m")
    for worst in worst_names_data0:
        display_by_name(best,path_versions)
    print("\033[1mWorst libfmp:\033[0m")
    for worst in worst_names_data1:
        display_by_name(best,path_versions)
    print("")  


