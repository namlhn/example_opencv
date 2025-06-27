# main_app.py (Corrected and Improved Version 2)
#
# A comprehensive, interactive application for demonstrating key computer vision concepts.
# This version fixes a TypeError in the panorama stitching feature matching.

import cv2
import numpy as np
import os

def draw_text(frame, text, pos=(20, 50), scale=1.0, color=(255, 255, 255)):
    """Draws white text with a black outline for better visibility."""
    # The thickness is scaled with the font size for a consistent look
    thickness = int(scale * 2)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

class ComputerVisionDemo:

    TRACKBARS_CONFIG = {
        'Kernel': ('Kernel', 1, 20, 5),
        'Threshold_01': ('Threshold 1', 0, 255, 50),
        'Threshold_02': ('Threshold 2', 0, 255, 150),

    }
    TRACEBARS_CONFIG_GEOMETRY = {
        'Angle': ('Angle', -180, 180, 180), 
        'Translate X': ('Translate X', -150, 150, 150), 
        'Translate Y': ('Translate Y', -100, 100, 100), 
        'Scale': ('Scale', 20, 200, 100), 
    }
    CHESSBOARD_CONFIG = {
        'Rows': ('Rows', 4, 12, 8),      # Internal corners (rows)
        'Cols': ('Cols', 4, 12, 6),      # Internal corners (cols)
    }
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")

        # --- General State ---
        self.mode = 'NORMAL'
        self.debug_mode = False
        self.window_name = 'CV Exam ~ Le Hoai Nam ~ st125591'
        self.filter_type = 0  # 0: None, 1: Gray, 2: Blur, 3: Canny
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

        self.trackbars_created = False
        self.chessboard_detected = False
        self.chessboard_corners = None
        self.pano_images = []

    def _cleanup_ui(self):
        cv2.destroyWindow(self.window_name)
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        self.trackbars_created = False
        
        self.pano_images = []
        self.chessboard_detected = False
        self.chessboard_corners = None

    def _set_mode(self, new_mode):
        if self.mode != new_mode:
            self.mode = new_mode
            print(f"Switched to {self.mode} mode.")
            self._cleanup_ui()

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)

            display_frame = self._process_frame(frame.copy())

            help_text = "[N]ormal [Q]uit [F]ilter [G]eometry [P]ano [C]hessboard"
            draw_text(display_frame, help_text, pos=(10, display_frame.shape[0] - 20), scale=0.6)
            cv2.imshow(self.window_name, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if self._handle_key_press(key, frame):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

    def _process_frame(self, frame):
        mode_map = {
            'NORMAL': self._draw_normal_mode,
            'FILTER': self._draw_filter_mode,
            'GEOMETRY': self._draw_geometry_mode,
            'PANO': self._draw_pano_mode,  
            'CHESSBOARD': self._draw_chessboard_mode,
        }
        handler = mode_map.get(self.mode, self._draw_normal_mode)
        return handler(frame)
    
    def _draw_normal_mode(self, frame):
        draw_text(frame, "Mode: NORMAL")
        return frame
    
    # filter mode
    def _draw_filter_mode(self, frame):
      
        if self.filter_type == 0:
            pass
        elif self.filter_type == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif self.filter_type == 2:
            dummy_callback = lambda x: None
            name, min_val, max_val, default_pos = self.TRACKBARS_CONFIG['Kernel']
            if not self.trackbars_created:
                range_max = max_val if min_val == 0 else max_val - min_val
                cv2.createTrackbar(name, self.window_name, default_pos, range_max, dummy_callback)
                self.trackbars_created = True
            kernel_size = cv2.getTrackbarPos(name, self.window_name)
            if kernel_size % 2 == 0:  # Ensure kernel size is odd
                kernel_size += 1
            if kernel_size < 1: kernel_size = 1
            frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
        elif self.filter_type == 3:
            dummy_callback = lambda x: None
            name, _, max_val, default_pos = self.TRACKBARS_CONFIG['Threshold_01']
            name2, _, max_val2, default_pos2 = self.TRACKBARS_CONFIG['Threshold_02']
            if not self.trackbars_created:
                cv2.createTrackbar(name, self.window_name, default_pos, max_val, dummy_callback)
                cv2.createTrackbar(name2, self.window_name, default_pos2, max_val2, dummy_callback)
                self.trackbars_created = True
            threshold1 = cv2.getTrackbarPos(name, self.window_name)
            threshold2 = cv2.getTrackbarPos(name2, self.window_name)
            if threshold1 < 0: threshold1 = 0
            if threshold2 < 0: threshold2 = 0
            if threshold1 > 255: threshold1 = 255
            if threshold2 > 255: threshold2 = 255
          
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.Canny(img_gray, threshold1, threshold2, L2gradient= 0)

        draw_text(frame, "Mode: FILTER")
        draw_text(frame, "[0] NONE [1] GRAY [2] BLUR [3] CANDY", pos=(10, frame.shape[0] - 60),  scale=0.6)

        return frame

    def _draw_geometry_mode(self, frame):
       
        if not self.trackbars_created:
            for name, (label, min_val, max_val, default_pos) in self.TRACEBARS_CONFIG_GEOMETRY.items():
                cv2.createTrackbar(name, self.window_name, default_pos, max_val - min_val, lambda x: None)
            self.trackbars_created = True

        # Get trackbar positions
        angle_pos = cv2.getTrackbarPos('Angle', self.window_name)
        translate_x = cv2.getTrackbarPos('Translate X', self.window_name)
        translate_y = cv2.getTrackbarPos('Translate Y', self.window_name)
        scale_pos = cv2.getTrackbarPos('Scale', self.window_name) 
        angle = angle_pos + self.TRACEBARS_CONFIG_GEOMETRY['Angle'][1]

        tx = translate_x + self.TRACEBARS_CONFIG_GEOMETRY['Translate X'][1]
        ty = translate_y + self.TRACEBARS_CONFIG_GEOMETRY['Translate Y'][1]
        # Apply transformations
        rows, cols, _ = frame.shape
        center = (cols / 2, rows / 2)
        scale = scale_pos / 100.0

        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += tx  # Translate X
        M[1, 2] += ty  # Translate Y
        # Apply the affine transformation
        frame = cv2.warpAffine(frame, M, 
        (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=(47, 53, 66))

        draw_text(frame, "Mode: GEOMETRY")
        
        return frame
    
    def _draw_pano_mode(self, frame):
        draw_text(frame, "Mode: PANO")
        draw_text(frame, "[S] Save Panorama [C] Capture", pos=(10, frame.shape[0] - 60), scale=0.6)
        return frame
    
    def _handle_key_press(self, key, frame):
        if key == ord('q'): return True
        
        mode_keys = {
            ord('n'): 'NORMAL',
            ord('f'): 'FILTER',
            ord('g'): 'GEOMETRY',
            ord('p'): 'PANO',
            ord('c'): 'CHESSBOARD',
        }
        if key in mode_keys:
            self._set_mode(mode_keys[key])
            return False

        if key == ord('d'):
            self.debug_mode = not self.debug_mode
            print(f"Debug mode {'ON' if self.debug_mode else 'OFF'}")
            if not self.debug_mode:
                try: cv2.destroyWindow('Debug: Matches')
                except cv2.error: pass
            return False

        # Mode-specific keys
        if self.mode == 'FILTER':
            if key == ord('0'):
                self._cleanup_ui()
                self.filter_type = 0
            elif key == ord('1'):
                self._cleanup_ui()
                self.filter_type = 1
            elif key == ord('2'):
                self._cleanup_ui()
                self.filter_type = 2
            elif key == ord('3'):
                self._cleanup_ui()
                self.filter_type = 3
            return False
        

        if self.mode == 'PANO':
            if key == ord('s'):
                if len(self.pano_images) > 1:
                    print("Begin stitching panorama...")
                    stitcher = cv2.Stitcher_create()
                    status, result = stitcher.stitch(self.pano_images)
                    if status == cv2.Stitcher_OK:
                        pano_path = 'panorama_result.jpg'
                        cv2.imwrite(pano_path, result)
                        # show panorama result
                        cv2.imshow('Panorama Result', result)
                        print(f"Panorama saved to {pano_path}")
                    else:
                        print("Error during panorama stitching:", status)
            elif key == ord('c'):
                print("Capturing panorama...")
                self.pano_images.append(frame.copy())
                print("Total captured images:", len(self.pano_images))
            return False
        if self.mode == 'CHESSBOARD':
            if key == ord('s') and self.chessboard_detected:
                filename = f'chessboard_corners_{cv2.getTickCount()}.jpg'
                cv2.imwrite(filename, frame)
                print(f"Corner detection saved as {filename}")
                print(f"Total corners detected: {len(self.chessboard_corners)}")
            return False
        return False


    def _detect_chessboard(self, frame):
        try:
            rows = cv2.getTrackbarPos('Rows', self.window_name)
            cols = cv2.getTrackbarPos('Cols', self.window_name)
        except:
            rows, cols = 9, 6 
        
        # Ensure minimum values
        if rows < 4: rows = 4
        if cols < 4: cols = 4
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find chessboard corners
        chessboard_size = (cols, rows)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        
        if ret:
            # Refine corner positions for sub-pixel accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            self.chessboard_detected = True
            self.chessboard_corners = corners
            return True, corners, chessboard_size
        else:
            self.chessboard_detected = False
            self.chessboard_corners = None
            return False, None, chessboard_size

    def _draw_corner_points(self, frame, corners):
        """Draw individual corner points with numbers"""
        for i, corner in enumerate(corners):
            x, y = corner.ravel()
            x, y = int(x), int(y)
            
            # Draw corner point as circle
            cv2.circle(frame, (x, y), 8, (0, 255, 0), -1)  # Green filled circle
            #cv2.circle(frame, (x, y), 10, (0, 0, 255), 2)   # Red border
            
            # Draw corner number (every 5th corner to avoid clutter)
            """
            if i % 5 == 0:
                cv2.putText(frame, str(i), (x+12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            """
    def _draw_chessboard_mode(self, frame):
        if not self.trackbars_created:
            for name, (label, min_val, max_val, default_pos) in self.CHESSBOARD_CONFIG.items():
                cv2.createTrackbar(name, self.window_name, default_pos, max_val, lambda x: None)
            self.trackbars_created = True

        # Detect chessboard
        detected, corners, chessboard_size = self._detect_chessboard(frame)
        
        if detected:
            # Draw standard chessboard corners visualization
            cv2.drawChessboardCorners(frame, chessboard_size, corners, True)
            
            # Draw individual corner points
            self._draw_corner_points(frame, corners)
            
            # Display detection info
            rows, cols = chessboard_size[1], chessboard_size[0]
            draw_text(frame, f"Chessboard Detected: {cols+1}x{rows+1}", pos=(20, 80), color=(0, 255, 0))
            draw_text(frame, f"Tracking {len(corners)} corners", pos=(20, 110), color=(0, 255, 0))

        else:
            draw_text(frame, "No Chessboard detect", pos=(20, 80), color=(0, 0, 255))
            #rows = cv2.getTrackbarPos('Rows', self.window_name) if self.trackbars_created else 8
            #cols = cv2.getTrackbarPos('Cols', self.window_name) if self.trackbars_created else 6
            #draw_text(frame, f"Looking for: {cols+1}x{rows+1} board", pos=(20, 110), color=(255, 255, 0))

        draw_text(frame, "Mode: CHESSBOARD TRACKING")
        draw_text(frame, "[S] Save Detection", pos=(10, frame.shape[0] - 60), scale=0.6)
        
        return frame

if __name__ == '__main__':
    app = ComputerVisionDemo()
    app.run()