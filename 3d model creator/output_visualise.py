import open3d as o3d

pcd = o3d.io.read_point_cloud("path/to/fused.ply")
o3d.visualization.draw_geometries([pcd])
