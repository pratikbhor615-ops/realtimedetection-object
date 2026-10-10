-- ========================================================
-- Real-Time Object Detection & Logging Platform
-- Assignment 5 - Database Schema Definition
-- Database Name: vision_platform
-- ========================================================

-- Create the database if it doesn't already exist
CREATE DATABASE IF NOT EXISTS vision_platform
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

-- Switch to vision_platform database
USE vision_platform;

-- Create detection_logs table
CREATE TABLE IF NOT EXISTS detection_logs (
    log_id INT AUTO_INCREMENT PRIMARY KEY COMMENT 'Unique identifier for each detection event',
    timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Date and time of detection',
    object_class VARCHAR(50) NOT NULL COMMENT 'Class name of detected object (e.g. person, cell phone)',
    confidence FLOAT NOT NULL COMMENT 'Confidence score between 0.00 and 1.00',
    bbox_x INT NOT NULL COMMENT 'Bounding box X coordinate (top-left)',
    bbox_y INT NOT NULL COMMENT 'Bounding box Y coordinate (top-left)',
    bbox_w INT NOT NULL COMMENT 'Bounding box width',
    bbox_h INT NOT NULL COMMENT 'Bounding box height',
    INDEX idx_timestamp (timestamp),
    INDEX idx_object_class (object_class)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='Stores real-time object detection events';

-- Sample verification query:
-- SELECT * FROM detection_logs ORDER BY log_id DESC LIMIT 10;
