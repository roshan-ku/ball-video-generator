# Ball Video Generator

A Python script that generates test videos featuring bouncing, shape-shifting balls with elastic collisions, 140+ shuffled background patterns, a centered timer, and a 3×3 grid overlay. Designed for testing video pipelines with various pixel formats, resolutions, and frame rates.

## Features

- **15 bouncing balls** with elastic wall/ball collisions
- **5 shape types** that cycle every 3 seconds: circle, triangle, rectangle, pentagon, hexagon
- **140+ background patterns** (gradients, checkerboards, stripes, dots, plasma, brick, sunburst, noise, etc.) shuffled and switched every 10 seconds
- **3×3 grid overlay** for visual alignment testing
- **Centered timer** with shadow for readability
- **Color metadata by format**:
   - YUV outputs use HDR signaling (BT.2020 + PQ/SMPTE 2084 + mastering display metadata)
   - GBR outputs use RGB signaling (`gbr` matrix with BT.709 primaries/transfer)
- **Configurable**: resolution, frame rate, duration, pixel format

## Requirements

- Python 3.6+
- OpenCV (`pip install opencv-python`)
- NumPy (`pip install numpy`)
- A static FFmpeg binary named `ffmpeg_static` in the same directory as the script

## Usage

```bash
python3 generate_ball_video.py [OPTIONS]
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--width` | 1920 | Output video width |
| `--height` | 1080 | Output video height |
| `--fps` | 30 | Frames per second |
| `--duration` | 43200 (12h) | Duration in seconds |
| `--pix_fmt` | yuv420p | FFmpeg pixel format for output |
| `-o`, `--output` | random_balls_9quadrants.mp4 | Output filename |

### Examples

```bash
# 15-minute 1080p video at 30fps with 10-bit 4:2:2
python3 generate_ball_video.py --duration 900 --fps 30 --pix_fmt yuv422p10le -o test_10bit.mp4

# 15-minute 4K video at 60fps with 12-bit GBR planar
python3 generate_ball_video.py --duration 900 --fps 60 --width 3840 --height 2160 --pix_fmt gbrp12le -o test_4k.mp4

# 15-minute 2K (QHD) video at 60fps
python3 generate_ball_video.py --duration 900 --fps 60 --width 2560 --height 1440 --pix_fmt gbrp12le -o test_2k.mp4
```

### Supported Pixel Formats

Any format supported by FFmpeg's libx265 encoder, including:

- `yuv420p` — 8-bit 4:2:0
- `yuv422p10le` — 10-bit 4:2:2
- `yuv422p12le` — 12-bit 4:2:2
- `yuv444p12le` — 12-bit 4:4:4
- `gbrp10le` — 10-bit GBR planar
- `gbrp12le` — 12-bit GBR planar

## How It Works

1. **Initialization**: Parses arguments, locates the `ffmpeg_static` binary, and spawns an FFmpeg subprocess with `pipe:0` (stdin) as input.

2. **Background generation**: Pre-generates 140 unique background patterns (solids, gradients, checkerboards, stripes, dots, rings, crosshatch, diamonds, zigzag, brick, sunburst, grid lines, plasma, noise, hex patterns) and shuffles them.

3. **Frame loop**: For each frame:
   - Selects the current background (switches every 10s)
   - Updates ball positions with velocity-based physics
   - Handles elastic ball-to-ball and ball-to-wall collisions
   - Cycles ball shapes every 3 seconds
   - Draws the 3×3 grid overlay
   - Renders a centered elapsed-time timer (HH:MM:SS.cc)
   - Writes the raw BGR24 frame to FFmpeg's stdin pipe

4. **Encoding**: FFmpeg encodes the raw frames using libx265 (ultrafast preset, CRF 20) into MP4.
   - YUV pixel formats are tagged as HDR (BT.2020/PQ)
   - GBR pixel formats are tagged as RGB/BT.709 (`colormatrix=gbr`)

## Performance

On a typical workstation:
- **1080p 30fps**: ~95-100 fps generation speed (~5 min for 15 min video)
- **1080p 60fps**: ~95-100 fps (~10 min for 15 min video)
- **2K 60fps**: ~60 fps (~15 min for 15 min video)
- **4K 30fps**: ~25-30 fps (~20 min for 15 min video)
- **4K 60fps**: ~25-30 fps (~40 min for 15 min video)

## License

MIT
