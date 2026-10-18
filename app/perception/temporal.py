"""Actor-neutral proposals with explicit gaps; no role or legal conclusion."""
from dataclasses import dataclass, field
import math


def iou(a,b):
    intersection=max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union else 0.


@dataclass
class Track:
    id: int
    last_sec: float
    box: tuple
    previous_sec: float | None = None
    previous_box: tuple | None = None


class Tracker:
    def __init__(self,max_gap_sec=1.,min_iou=.2):
        self.tracks={}
        self.next_id=0
        self.max_gap_sec,self.min_iou=max_gap_sec,min_iou
        self.last_time=None

    def update(self,timestamp,detections):
        if not math.isfinite(timestamp) or timestamp<0 or (self.last_time is not None and timestamp<=self.last_time):
            raise ValueError('Tracker timestamps must increase')
        self.last_time=timestamp
        self.tracks={i:t for i,t in self.tracks.items() if timestamp-t.last_sec<=self.max_gap_sec}
        people=[d for d in detections if d.label=='person']
        pairs=sorted(((iou(t.box,d.box),i,j) for i,t in self.tracks.items() for j,d in enumerate(people)),reverse=True)
        assigned_tracks,assigned_people=set(),set()
        current=[]
        for score,i,j in pairs:
            if score<self.min_iou:
                break
            if i in assigned_tracks or j in assigned_people:
                continue
            track=self.tracks[i]
            track.previous_box,track.previous_sec=track.box,track.last_sec
            track.box,track.last_sec=people[j].box,timestamp
            current.append(track)
            assigned_tracks.add(i); assigned_people.add(j)
        for j,person in enumerate(people):
            if j not in assigned_people:
                track=Track(self.next_id,timestamp,person.box)
                self.tracks[self.next_id]=track
                self.next_id+=1
                current.append(track)
        return current


@dataclass
class CandidateWindow:
    start_sec: float
    end_sec: float
    reasons: set[str] = field(default_factory=set)
    actor_ids: set[int] = field(default_factory=set)


def plan_reviews(duration,signals,*,window_sec=10.,periodic_sec=15.):
    """Keep periodic context even without movement; preserve every signal in a bounded window.

    Adjacent proposals can share frames, but never merge into a giant interval
    later represented by only a few centre images. Complexity is explicit.
    """
    if not all(math.isfinite(x) and x>0 for x in (duration,window_sec,periodic_sec)):
        raise ValueError('Invalid review plan')
    anchors=[]
    position=0.
    while position<duration:
        anchors.append((min(duration,position+window_sec/2),'periodic_context',None))
        position+=periodic_sec
    anchors.append((duration,'end_context',None))
    anchors.extend(signals)
    windows=[]
    for timestamp,reason,actor in sorted(anchors,key=lambda x:x[0]):
        if not math.isfinite(timestamp) or not 0<=timestamp<=duration:
            raise ValueError('Signal outside source timeline')
        # A candidate is covered only if it has context on both sides where available.
        start=max(0.,timestamp-window_sec/2)
        end=min(duration,start+window_sec)
        start=max(0.,end-window_sec)
        existing=next((w for w in windows if w.start_sec<=start and w.end_sec>=end),None)
        if existing is None:
            existing=CandidateWindow(start,end)
            windows.append(existing)
        existing.reasons.add(reason)
        if actor is not None:
            existing.actor_ids.add(actor)
    return windows


def motion_signal(track,threshold=.12):
    if track.previous_box is None:
        return False
    dt=track.last_sec-track.previous_sec
    a,b=track.box,track.previous_box
    delta=math.hypot((a[0]+a[2]-b[0]-b[2])/2,(a[1]+a[3]-b[1]-b[3])/2)
    return dt>0 and delta/dt>=threshold
