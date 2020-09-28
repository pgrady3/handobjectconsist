import open3d
from pathlib import Path


def subsample_objects():
    ho_dir = 'data/ho3dv2'

    for path in Path(ho_dir).rglob('textured_simple.obj'):
        full_path = str(path.resolve())
        out_file = str(path.parent) + '/textured_simple_2000.obj'
        print(out_file)
        mesh = open3d.io.read_triangle_mesh(full_path)
        mesh_simple = mesh.simplify_quadric_decimation(2000)  # Reduce number of triangles

        open3d.io.write_triangle_mesh(out_file, mesh_simple)


if __name__ == "__main__":
    subsample_objects()
