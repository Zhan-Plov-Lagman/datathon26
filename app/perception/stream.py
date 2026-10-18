"""Bounded streaming frames using the team's actual presentation timeline."""
from pathlib import Path
import math

from app.video.sampling import SampledFrame, VideoError, check_deadline, probe_video


def iter_batches(path, *, fps=3.0, batch_size=4, max_long_edge=960, deadline=None, info=None):
    import av
    from PIL import Image

    if not math.isfinite(fps) or not 0 < fps <= 60:
        raise ValueError('fps must be finite and in (0, 60]')
    if type(batch_size) is not int or not 1 <= batch_size <= 32:
        raise ValueError('batch_size must be in [1, 32]')
    if type(max_long_edge) is not int or not 32 <= max_long_edge <= 4096:
        raise ValueError('max_long_edge must be in [32, 4096]')
    info = info or probe_video(path, deadline)
    batch, next_target, previous_time, last_selected, final = [], 0.0, None, None, None

    def capture(frame, timestamp):
        image = frame.to_image()
        if info.rotation:
            image = image.rotate(info.rotation, expand=True)
        image.thumbnail((max_long_edge, max_long_edge), Image.Resampling.LANCZOS)
        return SampledFrame(round(timestamp, 6), image)

    try:
        with av.open(str(Path(path).resolve(strict=True))) as container:
            if not container.streams.video:
                raise VideoError('No video stream')
            stream = container.streams.video[0]
            stream.thread_type = 'AUTO'
            for frame in container.decode(stream):
                check_deadline(deadline)
                if frame.pts is None or frame.time_base is None:
                    raise VideoError('Frame has no presentation timestamp')
                timestamp = float(frame.pts * frame.time_base) - info.start_time_sec
                if timestamp < -1e-4:
                    continue
                timestamp = max(0.0, timestamp)
                if previous_time is not None and timestamp < previous_time - 1e-6:
                    raise VideoError('Non-monotonic presentation timestamps')
                previous_time, final = timestamp, frame
                if timestamp + 1e-6 >= next_target and (last_selected is None or timestamp > last_selected + 1e-6):
                    batch.append(capture(frame, timestamp))
                    last_selected = timestamp
                    next_target = (math.floor(timestamp * fps + 1e-6) + 1) / fps
                    if len(batch) == batch_size:
                        yield batch
                        batch = []
            check_deadline(deadline)
            if final is None:
                raise VideoError('No decodable frames')
            if info.duration_sec - previous_time > .5:
                raise VideoError('Video ends before declared duration')
            if last_selected is None or previous_time > last_selected + 1e-6:
                batch.append(capture(final, previous_time))
            if batch:
                yield batch
    except (VideoError, TimeoutError):
        raise
    except Exception as exc:
        raise VideoError('Video decoding failed') from exc
