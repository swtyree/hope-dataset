# import (additional, slower imports included after parsing command line arguments)
import json
import os
import argparse
import textwrap

# parse arguments
parser = argparse.ArgumentParser(
    description='\n'.join(textwrap.wrap("""
Display a scene from the HOPE-Image or HOPE-Video datasets.
By default, object annotations are overlaid on either the reconstructed scene point cloud or RGBD. An overlay on the RGB image can be shown with `--showrgb`. File paths are automatically attempted relative to the annotation file path.
Press O to toggle objects, D to toggle RGBD (if present), P to toggle scene point cloud (if present), and Q to quit.
""".strip(), width=80)),
    formatter_class=argparse.RawTextHelpFormatter
)

parser.add_argument('annotspath', metavar='PATH',
                    help='Path to scene annotation file')
parser.add_argument('--showrgb', action='store_true',
                    help='Show RGB image instead of RGBD and/or point cloud\n \n')
parser.add_argument('--rgbpath', default=None, metavar='PATH',
                    help='Path to RGB image\n(optional, default: annotspath.replace(".json","_rgb.jpg"))')
parser.add_argument('--depthpath', default=None, metavar='PATH',
                    help='Path to depth image\n(optional, default: annotspath.replace(".json","_depth.png"))')
parser.add_argument('--pcpath', default=None, metavar='PATH',
                    help='Path to scene point cloud\n(optional, default: dirname(annotspath)+"/scene.ply"))')
parser.add_argument('--meshdir', default='meshes/eval/', metavar='PATH',
                    help='Path to object meshes\n(optional, default: meshes/eval/)')
args = parser.parse_args()


# validate annotations path
if not os.path.exists(args.annotspath):
    raise FileNotFoundError(f'Unable to find annotations path {args.annotspath}')
    
# if showing 3D, ensure there is a valid 3D input
args.show3d = not args.showrgb
if args.show3d:
    if args.pcpath is not None and not os.path.exists(args.pcpath):
        raise FileNotFoundError(f'Unable to find scene point cloud {args.pcpath}')
    if args.depthpath is not None and not os.path.exists(args.depthpath):
        raise FileNotFoundError(f'Unable to find depth file {args.depthpath}')
    if args.pcpath is None and args.depthpath is None:
        args.pcpath = os.path.join(os.path.dirname(args.annotspath), 'scene.ply')
        if not os.path.exists(args.pcpath):
            args.depthpath = args.annotspath.replace('.json', '_depth.png')
            if not os.path.exists(args.depthpath):
                raise FileNotFoundError(f'Unable to find either scene point cloud {args.pcpath} or depth file {args.depthpath}')
            args.pcpath = None

# validate RGB path
if args.rgbpath is None and args.showrgb or args.depthpath is not None:
    args.rgbpath = args.annotspath.replace('.json', '_rgb.jpg')
    if not os.path.exists(args.rgbpath):
        raise FileNotFoundError(f'Unable to find RGB image path {args.rgbpath}')
else:
    args.rgbpath = None

# validate mesh directory path
if not os.path.exists(args.meshdir):
    raise FileNotFoundError(f'Unable to find mesh directory {args.meshdir}')


# import visualization tools
print('Importing visualization tools...')
try:
    import numpy as np
    import trimesh
    import pyglet
    import PIL
    if args.show3d:
        import open3d as o3d
except ModuleNotFoundError as e:
    print(f'Some required Python packages are missing: {e}.\nPlease run: `pip install numpy open3d trimesh networkx pyglet Pillow`\n')
    raise


# mesh loading function
def load_mesh(object_class):
    mesh_fn = os.path.join(args.meshdir, f'{object_class}.obj')
    if not os.path.exists(mesh_fn):
        raise FileNotFoundError(f'Unable to open mesh path {mesh_fn}')
    return trimesh.load(mesh_fn)


# load scene annotations
print(f'Loading annotations ({args.annotspath})...')
annots = json.load(open(args.annotspath))


# get camera intrinsics
camera_intrinsics = np.array(annots['camera']['intrinsics'])
w, h = annots['camera']['width'], annots['camera']['height']
fx, fy, _ = np.diag(camera_intrinsics)
cx, cy, _ = camera_intrinsics[:, -1]


# initialize scene
scene = trimesh.scene.Scene()
scene.camera.K = camera_intrinsics
scene.camera_transform = scene.camera_transform @ np.diag([1, -1,-1, 1])


# load scene point cloud
if args.pcpath is not None:
    print(f'Loading scene point cloud ({args.pcpath})...')
    camera_extrinsics = np.array(annots['camera']['extrinsics'])

    pcd = o3d.io.read_point_cloud(args.pcpath)

    pcd = pcd.voxel_down_sample(voxel_size=0.002)
    pcd=pcd.transform(camera_extrinsics)
    # convert point cloud to trimesh
    point_cloud = trimesh.PointCloud(
        vertices=np.asarray(pcd.points),
        colors=np.asarray(pcd.colors)
    )

    # scale and filter
    point_cloud.apply_scale(100) # m -> cm
    trimesh.tol.merge = 0.1; point_cloud.merge_vertices() # merge points within 1mm
    point_cloud_name = scene.add_geometry(point_cloud)


# load RGBD cloud
if args.depthpath is not None:
    print(f'Loading RGDB ({args.depthpath})...')
    
    # load color and depth images
    rgb_image = o3d.io.read_image(args.rgbpath)
    depth_image = o3d.io.read_image(args.depthpath)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        rgb_image, depth_image,
        depth_scale=1/0.0010000000474974513,
        convert_rgb_to_intensity=False
    )
    w,h,_ = np.array(rgb_image).shape

    # convert depth image to point cloud
    depth_point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        o3d.camera.PinholeCameraIntrinsic(
            w, h, fx, fy, cx, cy
        )
    )
    depth_point_cloud = depth_point_cloud.crop(
        o3d.geometry.AxisAlignedBoundingBox(
            [-1.0, -0.5, 0.0], [1.0, 1.0, 2.0] # in m
    ))

    # convert point cloud to trimesh
    depth = trimesh.PointCloud(
        vertices=np.asarray(depth_point_cloud.points),
        colors=np.asarray(depth_point_cloud.colors)
    )

    # scale and filter
    depth.apply_scale(100) # m -> cm
    trimesh.tol.merge = 0.1; depth.merge_vertices() # merge points within 1mm
    depth_name = scene.add_geometry(depth)
    window_resolution = w,h


# load RGB image
if args.showrgb:
    print(f'Loading RGB ({args.rgbpath})...')
    
    # add rectangle with RGB image
    rgb_image = PIL.Image.open(args.rgbpath)
    w,h = rgb_image.size
    image_mesh = trimesh.creation.box()

    uv = image_mesh.vertices[:,:2] + 0.5; uv[:,1] *= -1 # get uv for texturing
    z = 300 # cm

    image_mesh.apply_scale((w,h,0))
    image_mesh.apply_translation((w/2,h/2,0))
    vertices = image_mesh.vertices
    vertices[:,0] = (vertices[:,0] - cx) * z / fx # x
    vertices[:,1] = (vertices[:,1] - cy) * z / fy # y
    vertices[:,2] = z # z

    image_mesh.visual = trimesh.visual.texture.TextureVisuals(uv=uv, image=rgb_image)
    image_mesh_name = scene.add_geometry(image_mesh)
    window_resolution = w,h

# load objects
print(f'Loading object meshes (from {args.meshdir})...')
mesh_names = []
for obj in annots['objects']:
    mesh = load_mesh(obj['class'])
    transformation_matrix = np.array(obj['pose'])
    mesh.apply_transform(transformation_matrix)
    mesh_name = scene.add_geometry(mesh)
    mesh_names.append(mesh_name)

# static scene viewer class
print('Loading scene viewer...')
from trimesh.viewer import SceneViewer
class StaticSceneViewer(SceneViewer):
    def __init__(self, *args, **kwargs):
        self.is_static = kwargs.get('static', False)
        super().__init__(*args, **kwargs)

    def on_mouse_scroll(self, x, y, dx, dy):
        if self.is_static:
            pass
        else:
            super().on_mouse_scroll(x, y, dx, dy)

    def on_mouse_press(self, x, y, buttons, modifiers):
        if self.is_static:
            pass
        else:
            super().on_mouse_press(x, y, buttons, modifiers)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if self.is_static:
            pass
        else:
            super().on_mouse_drag(x, y, dx, dy, buttons, modifiers)


# initialize scene viewer
print('Initializing scene viewer...')
window = StaticSceneViewer(scene, start_loop=False, static=args.showrgb)
if args.show3d and args.pcpath is not None: scene.point_cloud_visible = True
if args.show3d and args.depthpath is not None: scene.depth_visible = True
scene.meshes_visible = True
scene.window = window


# toggle mesh and point cloud visibility
@window.event
def on_key_press(symbol, modifiers):
    # quit
    if symbol == pyglet.window.key.Q:
        window.close()

    # toggle mesh visibility
    if symbol == pyglet.window.key.O:
        if scene.meshes_visible:
            for mesh_name in mesh_names:
                window.hide_geometry(mesh_name)
        else:
            for mesh_name in mesh_names:
                window.unhide_geometry(mesh_name)
        scene.meshes_visible = not scene.meshes_visible

    # toggle RGBD visibility
    if args.depthpath and symbol == pyglet.window.key.D:
        if scene.depth_visible:
            window.hide_geometry(depth_name)
        else:
            window.unhide_geometry(depth_name)
        scene.depth_visible = not scene.depth_visible

    # toggle point cloud visibility
    if args.pcpath and symbol == pyglet.window.key.P:
        if scene.point_cloud_visible:
            window.hide_geometry(point_cloud_name)
        else:
            window.unhide_geometry(point_cloud_name)
        scene.point_cloud_visible = not scene.point_cloud_visible


# show scene
print('Running...')
pyglet.app.run()
print('Done.')
