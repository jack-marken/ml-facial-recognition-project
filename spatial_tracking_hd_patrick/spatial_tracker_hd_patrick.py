import time
import csv
import os
from datetime import datetime
import cv2
import numpy as np

class SpatialAttendanceTracker:
    def __init__(self, camera_frame_width, camera_frame_height):
        # Sets the width of the incoming webcam video feed.
        self.camera_frame_width = camera_frame_width
        
        # Sets the height of the incoming webcam video feed.
        self.camera_frame_height = camera_frame_height
        
        # Stores the movement and time data for every registered individual.
        self.session_tracking_database = {}

    def calculate_spatial_zone(self, bounding_box_x1, bounding_box_y1, bounding_box_x2, bounding_box_y2):
        # Calculate the exact center pixel coordinates of the detected face bounding box.
        face_center_x = int((bounding_box_x1 + bounding_box_x2) / 2)
        face_center_y = int((bounding_box_y1 + bounding_box_y2) / 2)

        # Determine the horizontal position by dividing the camera width into equal thirds.
        if face_center_x < (self.camera_frame_width / 3):
            horizontal_position = "LEFT"
        elif face_center_x < ((self.camera_frame_width / 3) * 2):
            horizontal_position = "CENTER"
        else:
            horizontal_position = "RIGHT"

        # Determine the vertical position by dividing the camera height into equal thirds.
        if face_center_y < (self.camera_frame_height / 3):
            vertical_position = "TOP"
        elif face_center_y < ((self.camera_frame_height / 3) * 2):
            vertical_position = "MIDDLE"
        else:
            vertical_position = "BOTTOM"

        # Concatenate the vertical and horizontal strings to create the final zone identifier.
        final_zone_identifier = f"{vertical_position}_{horizontal_position}"

        return final_zone_identifier, face_center_x, face_center_y
    
    def log_person(self, identity_name, bounding_box_coordinates):
        # Ignore unknown faces to prevent tracking unregistered individuals.
        if identity_name == "[unknown]":
            return False, "UNKNOWN"

        # Extract the individual coordinates from the bounding box list.
        bounding_box_x1 = bounding_box_coordinates[0]
        bounding_box_y1 = bounding_box_coordinates[1]
        bounding_box_x2 = bounding_box_coordinates[2]
        bounding_box_y2 = bounding_box_coordinates[3]

        # Calculate the current spatial zone and the exact center point of the face.
        current_zone_identifier, face_center_x, face_center_y = self.calculate_spatial_zone(
            bounding_box_x1, 
            bounding_box_y1, 
            bounding_box_x2, 
            bounding_box_y2
        )
        
        # Get the current system time in seconds.
        current_system_time = time.time()

        # Check if the person is being tracked for the very first time.
        if identity_name not in self.session_tracking_database:
            
            # Format the current time into a readable string for the CSV report.
            formatted_arrival_time = datetime.now().strftime("%H:%M:%S")
            
            # Create a new detailed record for the person.
            self.session_tracking_database[identity_name] = {
                "arrival_time_formatted": formatted_arrival_time,
                "last_seen_timestamp": current_system_time,
                "last_path_logged_timestamp": current_system_time,
                "current_zone": current_zone_identifier,
                "total_zone_changes": 0,
                "zone_dwell_times_seconds": {
                    "TOP_LEFT": 0.0, "TOP_CENTER": 0.0, "TOP_RIGHT": 0.0,
                    "MIDDLE_LEFT": 0.0, "MIDDLE_CENTER": 0.0, "MIDDLE_RIGHT": 0.0,
                    "BOTTOM_LEFT": 0.0, "BOTTOM_CENTER": 0.0, "BOTTOM_RIGHT": 0.0
                },
                "movement_path_coordinates": [(face_center_x, face_center_y)]
            }
            return True, current_zone_identifier

        # Retrieve the existing record for the tracked person.
        active_record = self.session_tracking_database[identity_name]

        # Calculate the exact time spent since the camera last saw the person.
        time_difference_since_last_frame = current_system_time - active_record["last_seen_timestamp"]
        
        # Only add the time if the box disappeared for less than 3 seconds.
        if time_difference_since_last_frame < 3.0:
            active_record["zone_dwell_times_seconds"][active_record["current_zone"]] += time_difference_since_last_frame
        else:
            # If they were gone for more than 3 seconds, inject a 'None' to break the heatmap line.
            active_record["movement_path_coordinates"].append(None)
            
        # Update the last seen timestamp to the current time.
        active_record["last_seen_timestamp"] = current_system_time

        # Check if the person has moved into a completely different grid zone.
        if active_record["current_zone"] != current_zone_identifier:
            active_record["current_zone"] = current_zone_identifier
            active_record["total_zone_changes"] += 1

        # Calculate the time passed since the last movement coordinate was saved.
        path_logging_time_difference = current_system_time - active_record["last_path_logged_timestamp"]
        
        # Save the new coordinate only if one full second has elapsed to prevent micro-movement spam.
        if path_logging_time_difference >= 1.0:
            active_record["movement_path_coordinates"].append((face_center_x, face_center_y))
            active_record["last_path_logged_timestamp"] = current_system_time

        # Return False to indicate this is an update to an existing person, not a new arrival.
        return False, current_zone_identifier
    
    def generate_reports(self):
        # Create a timestamp for the folder and files so nothing is ever overwritten.
        session_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # UPDATED: Create a dedicated folder for this specific tracking session inside the spatial_tracking directory.
        session_folder_path = f"reports/spatial_tracking/session_{session_timestamp}"
        os.makedirs(session_folder_path, exist_ok=True)
        
        csv_filename = f"{session_folder_path}/attendance_log.csv"
        
        # 1. GENERATE THE CSV ATTENDANCE LOG
        with open(csv_filename, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["Name", "Arrival Time", "Final Zone", "Total Zone Changes", "Most Used Zone"])
            
            for name, data in self.session_tracking_database.items():
                # Automatically calculate which zone they spent the highest amount of seconds in.
                most_used_zone = max(data["zone_dwell_times_seconds"], key=data["zone_dwell_times_seconds"].get)
                
                csv_writer.writerow([
                    name,
                    data["arrival_time_formatted"],
                    data["current_zone"],
                    data["total_zone_changes"],
                    most_used_zone
                ])
        print(f"CSV Report successfully saved to {csv_filename}")

        # 2. GENERATE THE HEATMAP VISUALIZATIONS
        for name, data in self.session_tracking_database.items():
            image_filename = f"{session_folder_path}/heatmap_{name}.png"
            
            # Create a blank white image canvas based on the exact camera resolution.
            blank_image = np.ones((self.camera_frame_height, self.camera_frame_width, 3), dtype=np.uint8) * 255
            
            # Draw the 3x3 grid lines in light grey.
            cv2.line(blank_image, (int(self.camera_frame_width/3), 0), (int(self.camera_frame_width/3), self.camera_frame_height), (200, 200, 200), 2)
            cv2.line(blank_image, (int(self.camera_frame_width/3 * 2), 0), (int(self.camera_frame_width/3 * 2), self.camera_frame_height), (200, 200, 200), 2)
            cv2.line(blank_image, (0, int(self.camera_frame_height/3)), (self.camera_frame_width, int(self.camera_frame_height/3)), (200, 200, 200), 2)
            cv2.line(blank_image, (0, int(self.camera_frame_height/3 * 2)), (self.camera_frame_width, int(self.camera_frame_height/3 * 2)), (200, 200, 200), 2)
            
            # Draw the movement path trace.
            coordinates_list = data["movement_path_coordinates"]
            for index in range(1, len(coordinates_list)):
                previous_point = coordinates_list[index-1]
                current_point = coordinates_list[index]
                
                # THE LASER BEAM FIX: If either point is None, someone left the camera view.
                # We skip drawing the connecting line to break the path.
                if previous_point is None or current_point is None:
                    # We still draw the dot if the current point is valid when they return.
                    if current_point is not None:
                        cv2.circle(blank_image, current_point, 4, (255, 0, 0), -1)
                    continue
                
                # If both points are valid, draw a red line connecting them.
                cv2.line(blank_image, previous_point, current_point, (0, 0, 255), 2)
                # Draw a blue dot at the exact coordinate.
                cv2.circle(blank_image, current_point, 4, (255, 0, 0), -1)
            
            # Save the final drawn image to the session folder.
            cv2.imwrite(image_filename, blank_image)
            print(f"Heatmap for {name} successfully saved to {image_filename}")