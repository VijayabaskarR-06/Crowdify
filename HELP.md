# Help & Troubleshooting

## Common Issues

### 1. Camera Not Detected
- Ensure your webcam is not being used by another application (e.g., Zoom, Teams).
- If using a laptop, ensure camera permissions are granted.

### 2. Low Performance
- The system uses CPU-based inference by default. For better speed, consider lowering the video resolution in `app.py`.

### 3. "Address already in use" Error
- This occurs if another instance of **Crowdify** is running. Use `lsof -i :5001` to find the process ID and `kill -9 <PID>` to stop it.

## Support
For more assistance, please refer to the [documentation](README.md).
