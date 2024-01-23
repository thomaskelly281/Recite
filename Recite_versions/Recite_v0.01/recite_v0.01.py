import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import csv
import mediapipe as mp
import warnings
import datetime
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

warnings.filterwarnings("ignore")

# Initialize CSV file to save landmarks
holistic_face_file = "Test_13_holistic_face.csv"
holistic_pose_file = "Test_13_holistic_pose.csv"
holistic_right_hand_file = "Test_13_holistic_right_hand.csv"
holistic_left_hand_file = "Test_13_holistic_left_hand.csv"
pose_csv_file = "Test_13_pose.csv"

# Variable to start with mediapipe off
pipe_holistic = [False]
pipe_pose = [False]

color_1 = '#A388F0'
color_2 = '#AE55F0'
color_3 = '#9C00EF'


# Rescales frames
def rescale_frame(frame, percent=20):
    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)


def mediapipe_holistic_toggle(button, pipe_running):
    """Checks if MediaPipe Holistic is running"""
    pipe_running[0] = not pipe_running[0]
    if pipe_running[0]:
        button.config(text="Stop MediaPipe")
    else:
        button.config(text="Start MediaPipe")


def mediapipe_pose_toggle(button, pipe_running):
    """Checks if MediaPipe Pose is running"""
    pipe_running[0] = not pipe_running[0]
    if pipe_running[0]:
        button.config(text="Stop MediaPipe")
    else:
        button.config(text="Start MediaPipe")


def write_holistic_landmarks_to_csv(face_row, pose_row, right_hand_row, left_hand_row, csv_file_face, csv_file_pose,
                                    csv_file_right_hand, csv_file_left_hand):
    """Writes Holistic landmarks to a CSV file. Landmarks are denoted as landmark.x1, landmark.y1, landmark.z1..."""
    with open(csv_file_face, 'a', newline='') as f:
        csv_wrote2 = csv.writer(f)
        # To add CSV column to beginning, add line below v
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        row = [timestamp]
        row.extend(face_row)
        csv_wrote2.writerow(row)

    with open(csv_file_pose, 'a', newline='') as f:
        csv_wrote2 = csv.writer(f)
        # To add CSV column to beginning, add line below v
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        row = [timestamp]
        row.extend(pose_row)
        csv_wrote2.writerow(row)

    with open(csv_file_right_hand, 'a', newline='') as f:
        csv_wrote2 = csv.writer(f)
        # To add CSV column to beginning, add line below v
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        row = [timestamp]
        row.extend(right_hand_row)
        csv_wrote2.writerow(row)

    with open(csv_file_left_hand, 'a', newline='') as f:
        csv_wrote2 = csv.writer(f)
        # To add CSV column to beginning, add line below v
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        row = [timestamp]
        row.extend(left_hand_row)
        csv_wrote2.writerow(row)


def write_pose_landmarks_to_csv(landmarks, csv_file):
    """Writes Pose landmarks to a CSV file. Landmarks are denoted as landmark.x1, landmark.y1, landmark.z1..."""
    with open(csv_file, 'a', newline='') as f:
        csv_wrote3 = csv.writer(f)
        # To add CSV column to beginning, add line below v
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        row = [timestamp]
        for landmark in landmarks.landmark:
            row.extend([landmark.x, landmark.y, landmark.z])
        csv_wrote3.writerow(row)


def make_holistic_csv_file():
    """Creates Holistic csv and header"""
    # Initialize header for CSV file
    # To add column name to header, add it to header_row, below v
    header_row1 = ["Time Stamp"]
    header_row2 = ["Time Stamp"]
    header_row3 = ["Time Stamp"]
    header_row4 = ["Time Stamp"]

    def make_face_holistic_csv():
        # MediaPipe Pose model has 33 landmarks, the first column is the date time
        for i in range(468):
            header_row1.extend([f"x{i}", f"y{i}", f"z{i}"])

        # Check if file exists. If exists, do not write new header
        file_exists = os.path.isfile('Test_13_holistic_face.csv')
        # Write header to CSV file
        if not file_exists:
            with open(holistic_face_file, 'w', newline='') as f:
                csv_wrote = csv.writer(f)
                csv_wrote.writerow(header_row1)

    def make_pose_holistic_csv():
        # MediaPipe Pose model has 33 landmarks, the first column is the date time
        for i in range(33):
            header_row2.extend([f"x{i}", f"y{i}", f"z{i}"])

        # Check if file exists. If exists, do not write new header
        file_exists = os.path.isfile('Test_13_holistic_pose.csv')
        # Write header to CSV file
        if not file_exists:
            with open(holistic_pose_file, 'w', newline='') as f:
                csv_wrote = csv.writer(f)
                csv_wrote.writerow(header_row2)

    def make_right_hand_holistic_csv():
        # MediaPipe Pose model has 33 landmarks, the first column is the date time
        for i in range(21):
            header_row3.extend([f"x{i}", f"y{i}", f"z{i}"])

        # Check if file exists. If exists, do not write new header
        file_exists = os.path.isfile('Test_13_holistic_right_hand.csv')
        # Write header to CSV file
        if not file_exists:
            with open(holistic_right_hand_file, 'w', newline='') as f:
                csv_wrote = csv.writer(f)
                csv_wrote.writerow(header_row3)

    def make_left_hand_holistic_csv():
        # MediaPipe Pose model has 33 landmarks, the first column is the date time
        for i in range(21):
            header_row4.extend([f"x{i}", f"y{i}", f"z{i}"])

        # Check if file exists. If exists, do not write new header
        file_exists = os.path.isfile('Test_13_holistic_left_hand.csv')
        # Write header to CSV file
        if not file_exists:
            with open(holistic_left_hand_file, 'w', newline='') as f:
                csv_wrote = csv.writer(f)
                csv_wrote.writerow(header_row4)

    make_face_holistic_csv()
    make_pose_holistic_csv()
    make_right_hand_holistic_csv()
    make_left_hand_holistic_csv()


def make_pose_csv_file():
    """Creates Pose csv and header"""
    # Initialize header for CSV file
    # To add column name to header, add it to header_row, below v
    header_row = ["Time Stamp"]
    for i in range(33):  # MediaPipe Pose model has 33 landmarks, the first column is the date time
        header_row.extend([f"x{i}", f"y{i}", f"z{i}"])

    # Check if file exists. If exists, do not write new header
    file_exists = os.path.isfile('Test_13_pose.csv')
    # Write header to CSV file
    if not file_exists:
        with open(pose_csv_file, 'w', newline='') as f:
            csv_wrote = csv.writer(f)
            csv_wrote.writerow(header_row)


# Initialize MediaPipe Holistic
mp_drawing = mp.solutions.drawing_utils  # Drawing Helpers
mp_holistic = mp.solutions.holistic  # MediaPipe Solutions
holistic = mp_holistic.Holistic()

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


def get_holistic_frame(cap):
    # Capture frame from webcam
    ret, frame = cap.read()

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = True

    # Process the frame and get the pose landmarks
    results = holistic.process(rgb_frame)

    rgb_frame.flags.writeable = False

    if results.face_landmarks:
        # Draw the landmarks on the frame
        mp_drawing.draw_landmarks(rgb_frame, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                                  mp_drawing.DrawingSpec(
                                      color=(80, 110, 10), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(
                                      color=(80, 256, 121), thickness=1, circle_radius=1)
                                  )
        # Extract Face Landmarks
        face = results.face_landmarks.landmark
        face_row = list(
            np.array([[landmark.x, landmark.y, landmark.z] for landmark in face]).flatten())
    else:
        face_row = []

    if results.right_hand_landmarks:
        # 2. Right hand
        mp_drawing.draw_landmarks(rgb_frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(
                                      color=(80, 22, 10), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(
                                      color=(80, 44, 121), thickness=2, circle_radius=2)
                                  )
        # Extract Right-Hand Landmarks
        right_hand = results.right_hand_landmarks.landmark
        right_hand_row = list(
            np.array([[landmark.x, landmark.y, landmark.z] for landmark in right_hand]).flatten())
    else:
        right_hand_row = []

    if results.left_hand_landmarks:
        # 3. Left Hand
        mp_drawing.draw_landmarks(rgb_frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                  mp_drawing.DrawingSpec(
                                      color=(121, 22, 76), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(
                                      color=(121, 44, 250), thickness=2, circle_radius=2)
                                  )
        # Extract Left-Hand Landmarks
        left_hand = results.left_hand_landmarks.landmark
        left_hand_row = list(
            np.array([[landmark.x, landmark.y, landmark.z] for landmark in left_hand]).flatten())
    else:
        left_hand_row = []

    if results.pose_landmarks:
        # 4. Pose Detections
        mp_drawing.draw_landmarks(rgb_frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(
                                      color=(245, 117, 66), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(
                                      color=(245, 66, 230), thickness=2, circle_radius=2)
                                  )
        # Extract Pose Landmarks
        pose_2 = results.pose_landmarks.landmark
        pose_row = list(
            np.array([[landmark.x, landmark.y, landmark.z] for landmark in pose_2]).flatten())
    else:
        pose_row = []

    write_holistic_landmarks_to_csv(face_row, pose_row, right_hand_row, left_hand_row, holistic_face_file,
                                    holistic_pose_file, holistic_right_hand_file, holistic_left_hand_file)

    return cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2RGBA)


def get_pose_frame(cap):
    # Capture frame from webcam
    ret, frame = cap.read()

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and get the pose landmarks
    results = pose.process(rgb_frame)

    # Draw the landmarks on the frame
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            rgb_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        write_pose_landmarks_to_csv(results.pose_landmarks, pose_csv_file)

    return cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2RGBA)


def blank_cam(cap):
    # Capture frame from webcam
    ret, frame = cap.read()

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2RGBA)


def update_image_label(cap, label, camera_running):
    if camera_running[0]:
        # Get the latest frame and convert it to ImageTk format
        if pipe_holistic != [False]:
            frame = get_holistic_frame(cap)
            # Calls frame rescaling
            if frame.shape[1] >= 900:
                frame = rescale_frame(frame, percent=20)
        elif pipe_pose != [False]:
            frame = get_pose_frame(cap)
            # Calls frame rescaling
            if frame.shape[1] >= 900:
                frame = rescale_frame(frame, percent=20)
        else:
            frame = blank_cam(cap)
            # Calls frame rescaling
            if frame.shape[1] >= 900:
                frame = rescale_frame(frame, percent=20)

        img = ImageTk.PhotoImage(image=Image.fromarray(frame))

        # Update the label with the new image
        label.config(image=img)
        label.image = img

        # Repeat after a short delay
        label.after(10, update_image_label, cap, label, camera_running)


def exit_app(root):
    root.destroy()
    print("You have quit the application")


# Data Visualisations
def plot_merged_scatterplot(df_cleaned, df2_cleaned, df3_cleaned):
    # Merge the three cleaned dataframes into a single dataframe, excluding df1_cleaned
    merged_df = pd.concat([df_cleaned, df2_cleaned, df3_cleaned],
                          keys=['df_cleaned', 'df2_cleaned', 'df3_cleaned'], axis=1)

    # Drop any rows where all elements are NaN across all dataframes
    merged_df.dropna(how='all', inplace=True)

    # Create a function to plot XYZ coordinates
    def plot_coordinates(df, color='purple'):
        for i in range(0, df.shape[1], 3):
            # Drop rows where any element is NaN for the current x, y, z
            subset = df.iloc[:, i:i + 3].dropna()
            if subset.empty:
                continue

            x = subset.iloc[:, 0]
            y = subset.iloc[:, 1]
            z = subset.iloc[:, 2]

            # Reverse normalize z for opacity (higher opacity for points closer to the camera)
            z_normalized = 1 - ((z - z.min()) / (z.max() - z.min()))

            # Scatter plot using x and y coordinates, with reversed z as opacity
            plt.scatter(x, y, c=color, alpha=z_normalized,
                        label=f'Point_{i // 3}')

    # Create a single plot
    plt.figure(figsize=(10, 14))

    # Colors for each original DataFrame, excluding df1_cleaned
    colors = {'df_cleaned': color_1,
              'df2_cleaned': color_2, 'df3_cleaned': color_3}

    # Loop through each set of columns corresponding to each original DataFrame and plot, excluding df1_cleaned
    for key, color in colors.items():
        subset_df = merged_df[key]
        plot_coordinates(subset_df, color=color)

    plt.title("Coordinates for Merged Cleaned DataFrames")
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.ylim(plt.ylim()[::-1])  # Flip the y-axis only once after all plotting
    # plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


def plot_smaller_scatterplot(df, df1, df2, df3):
    """Plots a scatterplot of x,y and z points of all 4 DataFrames, but plot only shows 10 rows, even distributed
     between the 1st and last row of the DataFrame"""
    # Merge the three cleaned dataframes into a single dataframe, excluding df1_cleaned
    merged_df = pd.concat([df, df1, df2, df3],
                          keys=['df', 'df1', 'df2', 'df3'], axis=1)

    # Function to plot selected rows in 2D scatter plot with normalized opacity based on z-coordinate
    def plot_selected_rows_2d_normalized(ax, df, color, marker, label_prefix):
        for idx, row in df.iterrows():
            xs = [row[i] for i in df.columns if 'x' in i]
            ys = [row[i] for i in df.columns if 'y' in i]
            zs = [row[i] for i in df.columns if 'z' in i]
            for x, y, z in zip(xs, ys, zs):
                ax.scatter(x, y, color=color, marker=marker, alpha=z,
                           label=f"{label_prefix} (Frame {idx})")

    # Drop any rows where all elements are NaN across all dataframes
    merged_df.dropna(how='all', inplace=True)

    # Create a function to plot XYZ coordinates
    def plot_coordinates(df, color='blue'):
        for i in range(0, df.shape[1], 3):
            # Drop rows where any element is NaN for the current x, y, z
            subset = df.iloc[:, i:i + 3].dropna()
            if subset.empty:
                continue

            x = subset.iloc[:, 0]
            y = subset.iloc[:, 1]
            z = subset.iloc[:, 2]

            # Reverse normalize z for opacity (higher opacity for points closer to the camera)
            z_normalized = 1 - ((z - z.min()) / (z.max() - z.min()))

            # Scatter plot using x and y coordinates, with reversed z as opacity
            plt.scatter(x, y, c=color, alpha=z_normalized,
                        label=f'Point_{i // 4}')

    # Create a single plot
    plt.figure(figsize=(10, 14))

    # Colors for each original DataFrame, excluding df1_cleaned
    colors = {'df': color_1, 'df1': 'purple', 'df2': color_2, 'df3': color_3}

    # Loop through each set of columns corresponding to each original DataFrame and plot, excluding df1_cleaned
    for key, color in colors.items():
        subset_df = merged_df[key]
        plot_coordinates(subset_df, color=color)

    plt.title("Coordinates for Merged Cleaned DataFrames")
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.ylim(plt.ylim()[::-1])  # Flip the y-axis only once after all plotting
    # plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


def interactive_scatter_plot(df_cleaned, df2_cleaned, df3_cleaned):
    # Merge the three cleaned dataframes into a single dataframe
    merged_df = pd.concat([df_cleaned, df2_cleaned, df3_cleaned],
                          keys=['df_cleaned', 'df2_cleaned', 'df3_cleaned'], axis=1)

    # Drop any rows where all elements are NaN across all dataframes
    merged_df.dropna(how='all', inplace=True)

    # Colors for each original DataFrame
    colors = {'df_cleaned': color_1,
              'df2_cleaned': color_2, 'df3_cleaned': color_3}

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(7, 8))
    plt.subplots_adjust(bottom=0.2)  # Make room for the slider

    def plot_coordinates(row=0):
        ax.clear()

        for key, color in colors.items():
            subset_df = merged_df[key].iloc[row].dropna()

            for i in range(0, len(subset_df), 3):
                x, y, z = subset_df[i:i + 3]

                # Reverse normalize z for opacity (higher opacity for points closer to the camera)
                z_normalized = 1 - ((z - subset_df.min()) /
                                    (subset_df.max() - subset_df.min()))

                # Scatter plot using x and y coordinates, with reversed z as opacity
                ax.scatter(x, y, c=color, alpha=z_normalized,
                           label=f'Point_{i // 3} ({key})')

        ax.set_title(
            f"Coordinates for Merged Cleaned DataFrames at Time {row}")
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_ylim(ax.get_ylim()[::-1])  # Flip the y-axis
        # ax.legend(loc='upper right')

    # Initial plot
    plot_coordinates(0)

    # Create the slider
    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, 'Row', 0, len(
        merged_df) - 1, valinit=0, valstep=1)

    # Update the plot when the slider is changed
    slider.on_changed(lambda val: plot_coordinates(int(val)))

    plt.show()


def plot_deviations(data, subplot_num, title):
    mean_values = data.mean()

    # Calculate the mean of each set of coordinates (x, y, z)
    coordinate_means = []
    for i in range(0, len(mean_values), 3):  # Iterate in steps of 3 for x, y, z
        mean_xyz = mean_values[i:i + 3].mean()
        coordinate_means.append(mean_xyz)

    # Calculate the deviation of each set of coordinates from their mean value over time
    deviations = []
    for i in range(0, data.shape[1], 3):  # Iterate in steps of 3 for x, y, z
        xyz = data.iloc[:, i:i + 3]
        deviations.append(xyz.mean(axis=1) - coordinate_means[i // 3])

    # Convert deviations list to DataFrame for easier plotting
    deviations_df = pd.concat(deviations, axis=1)
    deviations_df.columns = [
        f"Point_{i // 3}" for i in range(0, data.shape[1], 3)]

    # Plot the deviations over time in the specified subplot
    plt.subplot(2, 2, subplot_num)
    for col in deviations_df.columns:
        plt.plot(deviations_df[col], label=col)

    plt.axhline(0, color='black', linestyle='--')  # line for mean
    plt.title(title)
    plt.xlabel("Time (Rows)")
    plt.ylabel("Deviation from Mean Value")
    plt.grid(True)


def plot_all_deviations(df, df1, df2, df3):
    # Create a 2x2 grid for the subplots
    plt.figure(figsize=(14, 7))

    # Plot the deviations for each DataFrame
    plot_deviations(df, 1, "Deviation for Pose")
    plot_deviations(df1, 2, "Deviation for Face")
    plot_deviations(df2, 3, "Deviation for Right Hand")
    plot_deviations(df3, 4, "Deviation for Left Hand")

    plt.tight_layout()
    plt.show()


make_holistic_csv_file()
make_pose_csv_file()

# Load the CSV file into a Pandas DataFrame
csv_path = 'Test_13_holistic_pose.csv'
df = pd.read_csv(csv_path)

csv_path = 'Test_13_holistic_face.csv'
df1 = pd.read_csv(csv_path)

csv_path = 'Test_13_holistic_right_hand.csv'
df2 = pd.read_csv(csv_path)

csv_path = 'Test_13_holistic_left_hand.csv'
df3 = pd.read_csv(csv_path)

df = df.drop(columns=['Time Stamp'])
df1 = df1.drop(columns=['Time Stamp'])
df2 = df2.drop(columns=['Time Stamp'])
df3 = df3.drop(columns=['Time Stamp'])

# Drop rows containing NaN values from each DataFrame
df_cleaned = df.dropna()
df1_cleaned = df1.dropna()
df2_cleaned = df2.dropna()
df3_cleaned = df3.dropna()


# Main to run
def main():
    # Initialize webcam and set up the main window
    cap = cv2.VideoCapture(1, apiPreference=cv2.CAP_AVFOUNDATION)

    # Change resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 100)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 100)

    root = tk.Tk()
    root.title("Recite v0.1")
    root.configure(background="MediumPurple1")

    my_notebook = ttk.Notebook(root)
    my_notebook.pack(pady=15, padx=15)

    frame_1 = ttk.Frame(my_notebook)
    frame_2 = ttk.Frame(my_notebook)

    my_notebook.add(frame_1, text="MediaPipe")
    my_notebook.add(frame_2, text="Analysis")

    # Things being added to Frame 1 #
    # Create a label to hold the image
    label = tk.Label(frame_1)
    label.grid(padx=10, pady=30, row=1, column=2)

    # A flag to check if the camera is running
    # Using a list to make it mutable and allow changes inside functions
    camera_running = [True]

    # Button to toggle MediaPipe Pose Detection on and off
    button1 = tk.Button(frame_1, text="Start Holistic",
                        command=lambda: mediapipe_holistic_toggle(
                            button1, pipe_holistic),
                        highlightbackground="MediumPurple1", fg="Black", bg="White", activebackground="White")
    button1.grid(padx=20, pady=20, row=2, column=1)

    # Button to toggle MediaPipe Pose Detection on and off
    button2 = tk.Button(frame_1, text="Start Pose", command=lambda: mediapipe_pose_toggle(button2, pipe_pose),
                        highlightbackground="MediumPurple1", fg="Black", bg="White", activebackground="White")
    button2.grid(padx=20, pady=20, row=2, column=2)

    # Button will exit application
    button3 = tk.Button(frame_1, text="Exit", command=lambda: exit_app(root),
                        highlightbackground="MediumPurple1", fg="Black", bg="White", activebackground="White")
    button3.grid(padx=20, pady=20, row=2, column=3)

    update_image_label(cap, label, camera_running)

    # Things being added to Frame 2 #
    button_scatter = tk.Button(frame_2, text="Create Pose Scatterplot",
                               command=lambda: plot_merged_scatterplot(df_cleaned,
                                                                       df2_cleaned,
                                                                       df3_cleaned),
                               highlightbackground="MediumPurple1", fg="Black", bg="White", activebackground="White")
    button_scatter.grid(padx=20, pady=20, row=1, column=1)

    # Things being added to Frame 2 #
    button_small_scatter = tk.Button(frame_2, text="Create Smaller Scatterplot",
                                     command=lambda: plot_smaller_scatterplot(df,
                                                                              df1,
                                                                              df2,
                                                                              df3),
                                     highlightbackground="MediumPurple1", fg="Black", bg="White",
                                     activebackground="White")
    button_small_scatter.grid(padx=20, pady=20, row=1, column=2)

    button_inter_scatter = tk.Button(frame_2, text="Interactive Scatterplot",
                                     command=lambda: interactive_scatter_plot(df_cleaned,
                                                                              df2_cleaned,
                                                                              df3_cleaned),
                                     highlightbackground="MediumPurple1", fg="Black", bg="White",
                                     activebackground="White")
    button_inter_scatter.grid(padx=20, pady=20, row=1, column=3)

    button_deviations = tk.Button(frame_2, text="Plot Deviations",
                                  command=lambda: plot_all_deviations(
                                      df, df1, df2, df3),
                                  highlightbackground="MediumPurple1", fg="Black", bg="White",
                                  activebackground="White")
    button_deviations.grid(padx=20, pady=20, row=2, column=1)

    root.mainloop()

    # Release the webcam when closing the GUI
    cap.release()


main()
