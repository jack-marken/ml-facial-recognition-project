import time
import csv
import os
from datetime import datetime
import cv2
import numpy as np
import math

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
        
        # Only add the time if the box disappeared for less than 1.5 seconds.
        if time_difference_since_last_frame < 1.5:
            active_record["zone_dwell_times_seconds"][active_record["current_zone"]] += time_difference_since_last_frame
        else:
            # If they were gone for more than 1.5 seconds, inject a 'None' to break the heatmap line.
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
        
        # Create a dedicated folder for this specific tracking session inside the spatial_tracking directory.
        session_folder_path = f"reports/spatial_tracking/session_{session_timestamp}"
        os.makedirs(session_folder_path, exist_ok=True)
        
        csv_filename = f"{session_folder_path}/attendance_log.csv"
        
        # Generate the CSV Attendance Log
        with open(csv_filename, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["Name", "Arrival Time", "Final Zone", "Total Zone Changes", "Most Used Zone", "Total Seconds Tracked", "Total Distance (px)"])

            # Extract the dictionary items and sort them alphabetically by the identity_name key.
            sorted_tracking_data = sorted(self.session_tracking_database.items())
            
            for name, data in sorted_tracking_data:
                most_used_zone = max(data["zone_dwell_times_seconds"], key=data["zone_dwell_times_seconds"].get)
                total_time = sum(data["zone_dwell_times_seconds"].values())
                
                # Calculate the total physical distance moved across the camera frame in pixels.
                total_pixel_distance_moved = 0.0
                coordinate_list_length = len(data["movement_path_coordinates"])
                
                for current_index in range(1, coordinate_list_length):
                    previous_point = data["movement_path_coordinates"][current_index - 1]
                    current_point = data["movement_path_coordinates"][current_index]

                    # Verify both points contain valid coordinates before calculating distance.
                    if previous_point is not None and current_point is not None:
                        horizontal_difference = current_point[0] - previous_point[0]
                        vertical_difference = current_point[1] - previous_point[1]
                        distance_between_points = math.hypot(horizontal_difference, vertical_difference)
                        total_pixel_distance_moved = total_pixel_distance_moved + distance_between_points
                
                csv_writer.writerow([
                    name,
                    data["arrival_time_formatted"],
                    data["current_zone"],
                    data["total_zone_changes"],
                    most_used_zone,
                    round(total_time, 2),
                    round(total_pixel_distance_moved, 2)
                ])
        print(f"CSV Report successfully saved to {csv_filename}")

        # Generate the Colour-Coded Heatmap Visualisations
        for name, data in self.session_tracking_database.items():
            image_filename = f"{session_folder_path}/heatmap_{name}.png"
            
            # Create a blank white image canvas.
            blank_image = np.ones((self.camera_frame_height, self.camera_frame_width, 3), dtype=np.uint8) * 255
            
            # Define the exact pixel boundaries for the 9 zones to draw the color shading
            w_third = int(self.camera_frame_width / 3)
            h_third = int(self.camera_frame_height / 3)
            
            zones_mapping = {
                "TOP_LEFT": ((0, 0), (w_third, h_third)),
                "TOP_CENTER": ((w_third, 0), (w_third * 2, h_third)),
                "TOP_RIGHT": ((w_third * 2, 0), (self.camera_frame_width, h_third)),
                "MIDDLE_LEFT": ((0, h_third), (w_third, h_third * 2)),
                "MIDDLE_CENTER": ((w_third, h_third), (w_third * 2, h_third * 2)),
                "MIDDLE_RIGHT": ((w_third * 2, h_third), (self.camera_frame_width, h_third * 2)),
                "BOTTOM_LEFT": ((0, h_third * 2), (w_third, self.camera_frame_height)),
                "BOTTOM_CENTER": ((w_third, h_third * 2), (w_third * 2, self.camera_frame_height)),
                "BOTTOM_RIGHT": ((w_third * 2, h_third * 2), (self.camera_frame_width, self.camera_frame_height))
            }
            
            # Find the maximum time spent in a single zone to calculate the color scale.
            max_time = max(data["zone_dwell_times_seconds"].values())
            if max_time == 0:
                max_time = 1  # Set maximum time to 1 to prevent division by zero for new arrivals.
                
            # Shade the zones and print the time text
            for zone_name, (pt1, pt2) in zones_mapping.items():
                time_spent = data["zone_dwell_times_seconds"][zone_name]
                
                if time_spent > 0:
                    # Calculate color intensity (0 to 100). Higher time = deeper blue.
                    intensity = int((time_spent / max_time) * 100)
                    
                    # BGR Color Format coresponds to Blue, Green, Red. The blue channel remains constant while green and red are reduced to create a blue shade.
                    box_color = (255, 255 - intensity, 255 - intensity)
                    
                    # Fill the rectangle with the calculated color
                    cv2.rectangle(blank_image, pt1, pt2, box_color, -1)
                    
                    # Stamp the text in the top-left corner of each zone
                    text_x = pt1[0] + 10
                    text_y = pt1[1] + 30
                    cv2.putText(blank_image, f"{time_spent:.1f}s", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            # Draw the 3x3 grid lines in dark grey over the shading.
            cv2.line(blank_image, (w_third, 0), (w_third, self.camera_frame_height), (150, 150, 150), 2)
            cv2.line(blank_image, (w_third * 2, 0), (w_third * 2, self.camera_frame_height), (150, 150, 150), 2)
            cv2.line(blank_image, (0, h_third), (self.camera_frame_width, h_third), (150, 150, 150), 2)
            cv2.line(blank_image, (0, h_third * 2), (self.camera_frame_width, h_third * 2), (150, 150, 150), 2)
            
            # Draw the movement path trace (The Red lines and Blue Dots).
            coordinates_list = data["movement_path_coordinates"]
            for index in range(1, len(coordinates_list)):
                previous_point = coordinates_list[index-1]
                current_point = coordinates_list[index]
                
                if previous_point is None or current_point is None:
                    if current_point is not None:
                        cv2.circle(blank_image, current_point, 5, (255, 0, 0), -1)
                    continue
                
                cv2.line(blank_image, previous_point, current_point, (0, 0, 255), 2)
                cv2.circle(blank_image, current_point, 5, (255, 0, 0), -1)
            
            # Save the final drawn image to the session folder.
            cv2.imwrite(image_filename, blank_image)
            print(f"Heatmap for {name} successfully saved to {image_filename}")