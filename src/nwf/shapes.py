import numpy as np
from PIL import Image
from PIL import ImageDraw
from scipy.interpolate import make_interp_spline

from nwf import config


def circle(cx,cy,r):
    """Return the outline points of a circle as a polygon"""
    t = np.linspace(0,2*np.pi,180,endpoint=False)
    return np.stack([cx+r*np.cos(t),cy+r*np.sin(t)],axis=1)


def regular(n,rotation,inner=None):
    """Return a regular polygon or a star when an inner radius is given"""
    if inner is None:
        t = rotation+2*np.pi*np.arange(n)/n
        return np.stack([np.cos(t),np.sin(t)],axis=1)
    t = rotation+np.pi*np.arange(2*n)/n
    r = np.where(np.arange(2*n)%2==0,1.0,inner)
    return np.stack([r*np.cos(t),r*np.sin(t)],axis=1)


def heart():
    """Return the outline points of a heart"""
    t = np.linspace(0,2*np.pi,360,endpoint=False)
    x = 16*np.sin(t)**3
    y = 13*np.cos(t)-5*np.cos(2*t)-2*np.cos(3*t)-np.cos(4*t)
    return np.stack([x,y],axis=1)/16


# shapes are written in maths coordinates with y pointing up and every part is added or cut out in order
# the proportions follow the authors drawings so the triangle is as tall as it is wide and the oval is 0.61 as tall
NAMED = {
    "circle":[(circle(0,0,1),"add")],
    "oval":[(circle(0,0,1)*[1,0.61],"add")],
    "square":[(regular(4,np.pi/4),"add")],
    "diamond":[(regular(4,np.pi/2),"add")],
    "hexagon":[(regular(6,0),"add")],
    "pentagon":[(regular(5,np.pi/2),"add")],
    "triangle":[(np.array([[-1,-1],[1,-1],[0,1]]),"add")],
    "star":[(regular(5,np.pi/2,0.38),"add")],
}

ICONS = {
    "heart":[(heart(),"add")],
    "arrow":[(np.array([[-1,-0.25],[0.2,-0.25],[0.2,-0.6],[1,0],[0.2,0.6],[0.2,0.25],[-1,0.25]]),"add")],
    "crescent":[(circle(0,0,1),"add"),(circle(0.45,0.15,0.85),"cut")],
    "plus":[(np.array([[-0.33,-1],[0.33,-1],[0.33,-0.33],[1,-0.33],[1,0.33],[0.33,0.33],[0.33,1],[-0.33,1],[-0.33,0.33],[-1,0.33],[-1,-0.33],[-0.33,-0.33]]),"add")],
    "lightning":[(np.array([[0.2,1],[-0.5,0],[0,0],[-0.3,-1],[0.55,0.15],[0.05,0.15],[0.45,1]]),"add")],
    "cloud":[(circle(-0.55,-0.1,0.45),"add"),(circle(0,0.2,0.6),"add"),(circle(0.55,-0.05,0.45),"add"),(np.array([[-0.55,-0.55],[0.55,-0.55],[0.55,0],[-0.55,0]]),"add")],
    "key":[(circle(-0.6,0,0.4),"add"),(circle(-0.6,0,0.17),"cut"),(np.array([[-0.25,-0.08],[1,-0.08],[1,0.08],[-0.25,0.08]]),"add"),(np.array([[0.6,-0.3],[0.72,-0.3],[0.72,-0.08],[0.6,-0.08]]),"add"),(np.array([[0.85,-0.25],[0.97,-0.25],[0.97,-0.08],[0.85,-0.08]]),"add")],
    # the tip sits where the sides of the triangle touch the circle
    "teardrop":[(circle(0,-0.35,0.6),"add"),(np.array([[-0.52,-0.05],[0.52,-0.05],[0,0.85]]),"add")],
    "spade":[(heart()*[1,-1],"add"),(np.array([[0,-0.4],[-0.35,-1.2],[0.35,-1.2]]),"add")],
    "club":[(circle(0,0.45,0.42),"add"),(circle(-0.45,-0.15,0.42),"add"),(circle(0.45,-0.15,0.42),"add"),(circle(0,0.05,0.3),"add"),(np.array([[0,0],[-0.3,-1],[0.3,-1]]),"add")],
}


def squiggle(rng,anchors):
    """Draw anchor points with the authors random walk and return a smooth open spline through them"""
    centre = np.array([100.0,100.0])
    points = [centre]
    for _ in range(anchors):
        # the authors skip an anchor when 100 tries all fail so the same happens here
        for _ in range(100):
            angle = rng.uniform(0,2*np.pi)
            dist = rng.uniform(30,100)
            new = points[-1]+np.array([np.cos(angle)*dist,np.sin(angle)*dist])
            if np.linalg.norm(new-centre)<100 and all(np.linalg.norm(new-p)>=15 for p in points):
                points.append(new)
                break
    points = np.array(points)
    t = np.linspace(0,1,len(points))
    smooth = make_interp_spline(t,points,k=3)(np.linspace(0,1,len(points)*100))
    # flipping y because the authors worked in image coordinates and the drawing code expects y up
    return smooth*[1,-1]


def blob(rng):
    """Return a smooth closed random blob through eight anchors"""
    n = 8
    angles = 2*np.pi*(np.arange(n)+rng.uniform(-0.3,0.3,n))/n
    radii = rng.uniform(0.35,1.0,n)
    points = np.stack([radii*np.cos(angles),radii*np.sin(angles)],axis=1)
    # repeating the first anchor at the end so the periodic spline closes the curve
    points = np.vstack([points,points[:1]])
    t = np.linspace(0,1,n+1)
    return make_interp_spline(t,points,k=3,bc_type="periodic")(np.linspace(0,1,400,endpoint=False))


def draw(parts,mode):
    """Draw shape parts filled or as a closed outline or as an open line on a white square canvas"""
    size = config.CANVAS*config.SUPERSAMPLE
    # sizing by the added parts only because a cut part can reach past the visible shape
    allpoints = np.vstack([p for p,op in parts if op=="add"])
    low = allpoints.min(axis=0)
    high = allpoints.max(axis=0)
    scale = config.SHAPE_BOX*config.SUPERSAMPLE/(high-low).max()
    image = Image.new("RGB",(size,size),"white")
    pen = ImageDraw.Draw(image)
    for points,op in parts:
        # centring the bounding box and flipping y so that up in maths coordinates is up in the image
        xy = (points-(low+high)/2)*scale*[1,-1]+size/2
        xy = [tuple(p) for p in xy]
        if mode=="fill":
            pen.polygon(xy,fill=config.INK if op=="add" else "white")
        elif mode=="outline":
            pen.line(xy+xy[:1],fill=config.INK,width=config.STROKE*config.SUPERSAMPLE,joint="curve")
        else:
            pen.line(xy,fill=config.INK,width=config.STROKE*config.SUPERSAMPLE,joint="curve")
    return image.resize((config.CANVAS,config.CANVAS),Image.LANCZOS)
