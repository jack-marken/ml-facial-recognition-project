# ml-facial-recognition-project

## HD Spatial Tracking & Analytics
**Author:** Patrick Lunney 100599029  
**Branch:** `feature/Spatial-Tracking-HD-Patrick`

### What It Is
The HD Spatial Tracking feature is a monitoring tool built on top of the live face recognition UI, functioning similarly to a smart CCTV or physical security tracking system. While the base facial recognition system identifies *who* is in the frame, this spatial feature tracks *where* they are, *how long* they stay there, and *how far* they travel. It divides the camera frame into a 3x3 spatial grid and mathematically logs each user's movement session, ultimately exporting the data into visual heatmaps and detailed CSV attendance logs.

### Features
* **Real-Time Zone Tracking:** Dynamically calculates the exact centre of a user's bounding box and maps it to a 9-zone 3x3 grid (e.g., `TOP_LEFT`, `MIDDLE_CENTER`).
* **Live UI Integration:** Displays the user's current spatial zone directly beneath their name and similarity score on the live webcam feed.
* **Event-Based Verification**: To prevent the system from logging false data due to momentary misclassifications or spoofing flickers, the tracker utilises a 1-second pending queue. A user must be confidently and continuously recognised for 1 full second before an official tracking session begins, effectively filtering out transient glitches and empty heatmaps.
* **Distance Calculation:** Tracks the continuous coordinate path of the user and calculates the total physical distance travelled (in pixels) across the screen during the session.
* **Flicker & Movement Smoothing (1.5s Buffer):** When a person's face stops being tracked (e.g., due to motion blur or leaving the frame), the system waits 1.5 seconds before officially confirming they have exited. If they do not return within this window, their tracking path is intentionally broken. This prevents a massive, incorrect line from being drawn across the heatmap if an individual leaves the room and comes back later. Instead, it properly starts a second, separate tracking path.
* **Dynamic Heatmap Generation:** Automatically renders individual PNG heatmaps for every recognised person, colouring zones based on dwell time (deeper blue for longer duration) and plotting their movement path with red lines and blue coordinate dots.

### Files Updated & Created
* **`spatial_tracking_hd_patrick/spatial_tracker_hd_patrick.py`** *(New)*: Contains the core `SpatialAttendanceTracker` class, tracking logic, distance math, and the `generate_reports()` function.
* **`ui/user_interface.py`** *(Updated)*: Injected the tracker initialisation, updated the live OpenCV text renderer to include multi-line labels for zone identification, and added the end-of-session report generation trigger.

### Where The Outputs Go
All session data is saved locally on your machine in a timestamped folder automatically generated when the video window is closed. 

`reports/spatial_tracking/session_YYYY-MM-DD_HH-MM-SS/`

Inside each session folder, you will find:
1.  **`attendance_log.csv`**: A detailed spreadsheet acting as the primary data log. It records the exact time the person was first seen (Arrival Time), the last zone they were detected in (Final Zone), how many times they crossed into new grid areas (Total Zone Changes), the specific area they spent the most time in (Most Used Zone), their total time on camera (Total Seconds Tracked), and the estimated physical distance they moved across the frame (Total Distance in pixels).
2.  **`heatmap_[name].png`**: A tailored visual overlay generated for every individual recognised during the session. It features a 3x3 grid where each zone is shaded blue based on dwell time (the longer they stayed, the deeper the colour). It also plots their exact physical movement path using blue coordinate dots connected by red tracking lines, making it easy to visualise their exact route through the room.