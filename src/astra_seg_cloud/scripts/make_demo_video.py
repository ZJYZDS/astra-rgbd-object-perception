import numpy as np, sys, cv2
sys.path.insert(0,'/home/zjy/Astra_ws/src/astra_seg_cloud')
from rclpy.serialization import deserialize_message
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from sensor_msgs.msg import Image, PointCloud2
FX=FY=285.1711; CX,CY=159.5,119.5
BAG=sys.argv[1]; OUT=sys.argv[2]

def pc2_to_np(m):
    n=int(m.width)*int(m.height)
    if n==0: return None
    a=np.frombuffer(bytes(m.data),np.uint8)
    f=np.frombuffer(a,np.float32).reshape(n, m.point_step//4)
    return f[:, :4]          # x,y,z,rgb(packed)

r=SequentialReader(); r.open(StorageOptions(uri=BAG,storage_id='sqlite3'),ConverterOptions('cdr','cdr'))
ann={}; pcs={}
while r.has_next():
    t,d,ts=r.read_next()
    if t.endswith('/annotated'):
        m=deserialize_message(d,Image)
        ann[ts]=np.frombuffer(bytes(m.data),np.uint8).reshape(m.height,m.width,-1)[:,:,::-1].copy()
    elif t.endswith('/debug/all'):
        pcs[ts]=pc2_to_np(deserialize_message(d,PointCloud2))
keys=sorted(set(ann)&set(pcs))
print('可用帧',len(keys))
SC=3; W,H=320*SC,240*SC
vw=None
for k in keys:
    img=np.zeros((H,W,3),np.uint8)
    P=pcs[k]
    z=P[:,2]; ok=(z>0.05)&(z<3.0)
    u=(FX*P[ok,0]/z[ok]+CX)*SC; v=(FY*P[ok,1]/z[ok]+CY)*SC
    rgb=np.ascontiguousarray(P[ok,3]).view(np.uint32)
    col=np.stack([rgb&0xFF,(rgb>>8)&0xFF,(rgb>>16)&0xFF],1).astype(np.float32)
    # 提亮 + 点画 2x2，暗色物体（黑键盘）和平面点才看得见
    col=np.clip(col*2.0+45,0,255).astype(np.uint8)
    ui=np.clip(u.astype(int),1,W-2); vi=np.clip(v.astype(int),1,H-2)
    for dy in (0,1):
        for dx in (0,1):
            img[vi+dy,ui+dx]=col
    a=cv2.resize(ann[k],None,fx=SC,fy=SC,interpolation=cv2.INTER_NEAREST)
    h=max(a.shape[0],img.shape[0])
    left=np.zeros((h,a.shape[1],3),np.uint8); left[:a.shape[0]]=a
    right=np.zeros((h,W,3),np.uint8); right[:img.shape[0]]=img
    frame=np.hstack([left,right])
    cv2.putText(frame,'RGB + 3D bbox',(8,24),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(frame,'point cloud (all: plane+objects+box) + 3D bbox',(a.shape[1]+8,24),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2,cv2.LINE_AA)
    if vw is None:
        vw=cv2.VideoWriter(OUT,cv2.VideoWriter_fourcc(*'mp4v'),15.0,(frame.shape[1],frame.shape[0]))
    vw.write(frame)
vw.release(); print('写出',OUT)
