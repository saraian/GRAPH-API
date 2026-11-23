"""Scene-independent probes of RoomManager topology, cost and lock contention.

Run: python3 tools/analysis/probe_room_segmentation.py --output output/room_algorithm_audit
ROS interfaces are stubbed; numpy/OpenCV and all segmentation methods are real.
The concurrency probe models a bounded worker pool, NOT a real ROS executor.
"""
import argparse
import hashlib
import json
import platform
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace as NS

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'lost3dsg/src/perception_module'))
import rosstub
rosstub.install()
from room_manager import RoomManager


def manager():
    rm = RoomManager(node=None)
    rm._params.update(doorway_vlm_enabled=False, window_vlm_enabled=False)
    return rm


def grid(array, resolution):
    return NS(header=NS(frame_id='map'), data=np.where(array > 0, 0, 100).ravel(),
              info=NS(width=array.shape[1], height=array.shape[0], resolution=resolution,
                      origin=NS(position=NS(x=0., y=0., z=0.),
                                orientation=NS(x=0., y=0., z=0., w=1.))))


def two_rooms(res=.05, gap=0., noise=False):
    # Two 4 x 3 m rooms, separated by a 0.6 m wall and one 1 m doorway.
    px = lambda m: int(round(m/res))
    a = np.zeros((px(3.4), px(9.0)), np.uint8)
    a[px(.2):px(3.2), px(.2):px(8.8)] = 255
    a[px(.2):px(3.2), px(4.2):px(4.8)] = 0
    a[px(1.2):px(2.2), px(4.2):px(4.8)] = 255
    if gap:
        a[px(.5):px(.5)+max(1,px(gap)), px(4.2):px(4.8)] = 255
    if noise:
        # A small pocket inside the separator wall, connected through a neck
        # to the doorway. The doorway cut isolates it as a third component.
        a[px(.7):px(1.05), px(4.4):px(4.6)] = 255
        a[px(1.05):px(1.25), px(4.5)] = 255
    point = (px(1.7), px(4.5), 0., px(.5), .8)
    return a, point


def transformed(a, point, turns):
    y,x,theta,r,s = point
    for _ in range(turns):
        y,x = a.shape[1]-1-x,y
        a=np.rot90(a).copy()
        theta -= np.pi/2
    return a,(y,x,theta,r,s)


def cut_case(res, gap=0., noise=False, turns=0):
    rm=manager()
    try:
        a,p=two_rooms(res,gap,noise);a,p=transformed(a,p,turns)
        dist=cv2.distanceTransform(a,cv2.DIST_L2,5)
        started=time.perf_counter()
        out=rm._cut_free_space(a,dist,[p],res)
        return {'resolution_m':res,'gap_m':gap,'noise_pocket':noise,'rotation_deg':90*turns,
                'input_components':cv2.connectedComponents(a,8)[0]-1,
                'output_components':cv2.connectedComponents(out,8)[0]-1,
                'elapsed_ms':1000*(time.perf_counter()-started),
                'validation':rm._last_cut_stats}
    finally:
        rm._room_vlm_executor.shutdown()


def benchmark(size, candidates, repeat=3):
    rm=manager()
    try:
        # Scaling workload: unobstructed room with short non-splitting proposals.
        a=np.zeros((size,size),np.uint8);a[2:-2,2:-2]=255
        dist=np.ones_like(a,dtype=np.float32)*2
        n=int(np.ceil(np.sqrt(candidates)))
        coords=np.linspace(20,size-20,n,dtype=int)
        points=[(int(y),int(x),0.,2.,0.) for y in coords for x in coords][:candidates]
        timings=[]
        for _ in range(repeat):
            start=time.perf_counter();rm._cut_free_space(a,dist,points,.05)
            timings.append(1000*(time.perf_counter()-start))
        return {'grid_side_px':size,'candidates':candidates,'timings_ms':timings,
                'median_ms':float(np.median(timings)), 'accepted':rm._last_cut_stats['accepted']}
    finally:
        rm._room_vlm_executor.shutdown()


def full_pipeline(res,gap,noise=False):
    rm=manager()
    try:
        a,_=two_rooms(res,gap,noise)
        start=time.perf_counter();regions=rm._segment_regions_gvd(grid(a,res))
        return {'resolution_m':res,'gap_m':gap,'noise_pocket':noise,'regions':len(regions),
                'elapsed_ms':1000*(time.perf_counter()-start),
                'critical_points':rm.last_segmentation_stats.get('critical_points'),
                'cut_validation':rm.last_segmentation_stats.get('cut_validation')}
    finally:
        rm._room_vlm_executor.shutdown()


def lock_contention():
    rm=manager();entered=threading.Event();release=threading.Event()
    original=rm._segment_regions_gvd
    def controlled_segment(g):
        # Hold the real process_grid critical section deterministically. This
        # measures scheduling behaviour, not the duration of real segmentation.
        entered.set()
        if not release.wait(5): raise RuntimeError('probe release timeout')
        return original(g)
    rm._segment_regions_gvd=controlled_segment
    rm._schedule_window_vlm=lambda:None
    rm._publish_geometry=lambda g:None
    rm._save_rooms=lambda:None
    waiting=[threading.Event() for _ in range(3)]
    def camera(index):
        waiting[index].set()
        rm._camera_info_callback(NS())
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            f=pool.submit(rm.process_grid,grid(two_rooms()[0],.05))
            try:
                if not entered.wait(2): raise RuntimeError('segment did not start')
                other=[pool.submit(camera,i) for i in range(3)]
                for event in waiting:
                    if not event.wait(2): raise RuntimeError('callback did not start')
                bbox=pool.submit(lambda: 'bbox callback ran')
                time.sleep(.05)
                blocked=not bbox.done()
            finally:
                release.set()
            f.result(timeout=10)
            for callback in other: callback.result(timeout=10)
            result=bbox.result(timeout=10)
        return {'workers':4,'segment_holds_lock':True,'other_callbacks_waiting_on_lock':3,
                'unrelated_task_blocked':blocked,'task_runs_after_release':result,
                'scope':'synthetic bounded pool using real RoomManager callbacks; not ROS runtime evidence'}
    finally:
        release.set();rm._room_vlm_executor.shutdown()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    report={'environment':{'python':platform.python_version(),'opencv':cv2.__version__,
                           'opencv_threads':cv2.getNumThreads(),'machine':platform.machine(),
                           'room_manager_sha256':hashlib.sha256((ROOT/'lost3dsg/src/perception_module/room_manager.py').read_bytes()).hexdigest()},
            'cut_cases':[], 'full_pipeline':[], 'benchmarks':[]}
    rm=manager()
    report['parameters']=dict(rm._params)
    rm._room_vlm_executor.shutdown()
    for res in (.025,.05,.1):
        for turns in (0,1):
            for gap,noise in ((0.,False),(res,False),(.2,False),(0.,True)):
                report['cut_cases'].append(cut_case(res,gap,noise,turns))
        for gap in (0.,res,.2): report['full_pipeline'].append(full_pipeline(res,gap))
        report['full_pipeline'].append(full_pipeline(res,0.,True))
    for size in (200,400,800):
        for count in (1,8,32):report['benchmarks'].append(benchmark(size,count))
    report['lock_contention']=lock_contention()
    (args.output/'results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'cut_results':[(r['resolution_m'],r['rotation_deg'],r['gap_m'],r['noise_pocket'],r['output_components'],r['validation']['rejected_small_partition']) for r in report['cut_cases']],
                      'full_pipeline':[(r['resolution_m'],r['gap_m'],r['noise_pocket'],r['regions'],round(r['elapsed_ms'],1)) for r in report['full_pipeline']],
                      'benchmarks':[(r['grid_side_px'],r['candidates'],round(r['median_ms'],1)) for r in report['benchmarks']],
                      'lock_contention':report['lock_contention']},indent=2))

if __name__=='__main__':main()
