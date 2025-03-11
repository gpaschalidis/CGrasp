import numpy as np
import open3d as o3d
import os


def create_o3d_mesh(verts, faces, color):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.paint_uniform_color(color)
    mesh.compute_vertex_normals()
    return mesh


def create_line_set(origin, target_points, color, multiple_origins=False):
    points = np.concatenate((origin, target_points), 0)
    step = len(target_points) if multiple_origins else 1
    u = 1 if multiple_origins else 0
    lines = [[idp*u, idp+step] for idp in range(len(target_points))]
    colors = [color for k in range(len(lines))]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def read_o3d_mesh(path):
    return o3d.io.read_triangle_mesh(path)


