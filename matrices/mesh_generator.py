import gmsh
import meshio
import numpy as np
import argparse


def msh_to_xdmf(msh_file, xdmf_file=None):
    """
    Convert a .msh file to a .xdmf file compatible with dolfinx.

    Parameters
    ----------
    msh_file : str
        Path to the input .msh file.
    xdmf_file : str, optional
        Path to the output .xdmf file. Defaults to the same name with .xdmf extension.
    """
    if xdmf_file is None:
        xdmf_file = msh_file.rsplit(".", 1)[0] + ".xdmf"

    mesh = meshio.read(msh_file)

    # Extract only triangle cells
    triangle_cells = [c for c in mesh.cells if c.type == "triangle"]
    if not triangle_cells:
        raise ValueError(f"No triangle cells found in {msh_file}")

    # dolfinx expects (N, gdim) points; use 2D if all z == 0, else keep 3D
    points = mesh.points
    if points.shape[1] == 3 and np.allclose(points[:, 2], 0.0):
        points = points[:, :2]

    triangle_mesh = meshio.Mesh(
        points=points,
        cells=triangle_cells,
    )
    meshio.write(xdmf_file, triangle_mesh)
    print(f"Converted {msh_file} -> {xdmf_file}")


def generate_unit_square_mesh(n_segments, output_file="unit_square.xdmf"):
    """
    Generate a triangular mesh on a unit square [0,1]x[0,1].

    Parameters
    ----------
    n_segments : list of 4 ints
        Number of segments on each edge of the square:
        [bottom, right, top, left]
    output_file : str
        Path to the output .xdmf file.
    """
    gmsh.initialize()
    gmsh.model.add("unit_square")

    n_bottom, n_right, n_top, n_left = n_segments
    lc_bottom, lc_right, lc_top, lc_left = (1.0 / n for n in n_segments)

    # Corner points (z=0) — no characteristic length at points
    p1 = gmsh.model.geo.addPoint(0, 0, 0)  # bottom-left
    p2 = gmsh.model.geo.addPoint(1, 0, 0)  # bottom-right
    p3 = gmsh.model.geo.addPoint(1, 1, 0)  # top-right
    p4 = gmsh.model.geo.addPoint(0, 1, 0)  # top-left

    # Segments: bottom, right, top, left
    l_bottom = gmsh.model.geo.addLine(p1, p2)
    l_right  = gmsh.model.geo.addLine(p2, p3)
    l_top    = gmsh.model.geo.addLine(p3, p4)
    l_left   = gmsh.model.geo.addLine(p4, p1)

    # Curve loop and plane surface
    loop = gmsh.model.geo.addCurveLoop([l_bottom, l_right, l_top, l_left])
    surf = gmsh.model.geo.addPlaneSurface([loop])

    gmsh.model.geo.synchronize()

    # Background mesh fields: Distance + Threshold per edge, combined with Min
    field_id = 1
    threshold_fields = []
    lc_max = max(lc_bottom, lc_right, lc_top, lc_left)

    for line_tag, lc in [(l_bottom, lc_bottom), (l_right, lc_right),
                         (l_top, lc_top), (l_left, lc_left)]:
        # Distance field measuring distance to the edge
        dist_id = field_id
        gmsh.model.mesh.field.add("Distance", dist_id)
        gmsh.model.mesh.field.setNumbers(dist_id, "CurvesList", [line_tag])
        gmsh.model.mesh.field.setNumber(dist_id, "Sampling", max(100, int(1.0 / lc) * 10))
        field_id += 1

        # Threshold field: lc near the edge, transitions to lc_max further away
        thresh_id = field_id
        gmsh.model.mesh.field.add("Threshold", thresh_id)
        gmsh.model.mesh.field.setNumber(thresh_id, "InField", dist_id)
        gmsh.model.mesh.field.setNumber(thresh_id, "SizeMin", lc)
        gmsh.model.mesh.field.setNumber(thresh_id, "SizeMax", lc_max)
        gmsh.model.mesh.field.setNumber(thresh_id, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(thresh_id, "DistMax", 0.5)
        threshold_fields.append(thresh_id)
        field_id += 1

    # Combine all threshold fields with Min
    min_id = field_id
    gmsh.model.mesh.field.add("Min", min_id)
    gmsh.model.mesh.field.setNumbers(min_id, "FieldsList", threshold_fields)
    gmsh.model.mesh.field.setAsBackgroundMesh(min_id)

    # Disable mesh sizing from points and curvature, use only background field
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

    # Force triangular elements
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
    gmsh.option.setNumber("Mesh.ElementOrder", 1)

    # Generate 2D mesh
    gmsh.model.mesh.generate(2)

    # Show in gmsh GUI
    gmsh.fltk.run()

    # Write .msh, convert to .xdmf via meshio
    tmp_msh = output_file.rsplit(".", 1)[0] + ".msh"
    gmsh.write(tmp_msh)
    gmsh.finalize()

    msh_to_xdmf(tmp_msh, output_file)


def generate_unit_circle_mesh(lc_boundary, lc_center, output_file="unit_circle.xdmf"):
    """
    Generate a triangular mesh on a unit circle.

    Parameters
    ----------
    lc_boundary : float
        Target element size at the boundary (r = 1).
    lc_center : float
        Target element size at the center (r = 0).
    output_file : str
        Path to the output .xdmf file.
    """
    gmsh.initialize()
    gmsh.model.add("unit_circle")

    # Center point and circle curve via built-in kernel
    center = gmsh.model.occ.addPoint(0.0, 0.0, 0.0)
    gmsh.model.occ.addCircle(0.0, 0.0, 0.0, 1.0)   # curve tag 1
    gmsh.model.occ.addCurveLoop([1])                 # loop tag 1
    gmsh.model.occ.addPlaneSurface([1])              # surface tag 1

    gmsh.model.occ.synchronize()

    # Background field: Distance from center point + Threshold
    # Distance from the center point
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "PointsList", [center])

    # Threshold: lc_center at r=0, lc_boundary at r=1
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", lc_center)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", lc_boundary)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(2, "DistMax", 1.0)
    gmsh.model.mesh.field.setAsBackgroundMesh(2)

    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

    # Force triangular elements
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
    gmsh.option.setNumber("Mesh.ElementOrder", 1)

    gmsh.model.mesh.generate(2)

    # Show in gmsh GUI
    gmsh.fltk.run()

    # Write .msh, convert to .xdmf via meshio
    tmp_msh = output_file.rsplit(".", 1)[0] + ".msh"
    gmsh.write(tmp_msh)
    gmsh.finalize()

    msh_to_xdmf(tmp_msh, output_file)


def generate_annulus_mesh(hole_radius, lc_outer, lc_inner, output_file="annulus.xdmf"):
    """
    Generate a triangular mesh on an annulus (unit circle with concentric circular hole).

    Parameters
    ----------
    hole_radius : float
        Radius of the inner hole (must be in (0, 1)).
    lc_outer : float
        Target element size at the outer boundary (r = 1).
    lc_inner : float
        Target element size at the inner boundary (r = hole_radius).
    output_file : str
        Path to the output .xdmf file.
    """
    if not 0 < hole_radius < 1:
        raise ValueError("hole_radius must be strictly between 0 and 1")

    gmsh.initialize()
    gmsh.model.add("annulus")

    # Outer circle (r=1) and inner circle (r=hole_radius), both centered at origin
    gmsh.model.occ.addCircle(0.0, 0.0, 0.0, 1.0)           # curve tag 1
    gmsh.model.occ.addCircle(0.0, 0.0, 0.0, hole_radius)   # curve tag 2
    gmsh.model.occ.addCurveLoop([1])                        # outer loop, tag 1
    gmsh.model.occ.addCurveLoop([2])                        # inner loop, tag 2
    # Surface = outer disk minus inner disk (hole)
    gmsh.model.occ.addPlaneSurface([1, 2])                  # surface tag 1

    gmsh.model.occ.synchronize()

    # Distance field from the inner boundary curve
    gmsh.model.mesh.field.add("Distance", 1)
    gmsh.model.mesh.field.setNumbers(1, "CurvesList", [2])
    gmsh.model.mesh.field.setNumber(1, "Sampling", 200)

    # Threshold: lc_inner at the hole, transitions to lc_outer across the annulus width
    annulus_width = 1.0 - hole_radius
    gmsh.model.mesh.field.add("Threshold", 2)
    gmsh.model.mesh.field.setNumber(2, "InField", 1)
    gmsh.model.mesh.field.setNumber(2, "SizeMin", lc_inner)
    gmsh.model.mesh.field.setNumber(2, "SizeMax", lc_outer)
    gmsh.model.mesh.field.setNumber(2, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(2, "DistMax", annulus_width)

    # Distance field from the outer boundary curve
    gmsh.model.mesh.field.add("Distance", 3)
    gmsh.model.mesh.field.setNumbers(3, "CurvesList", [1])
    gmsh.model.mesh.field.setNumber(3, "Sampling", 200)

    # Threshold: lc_outer at the outer boundary, transitions inward
    gmsh.model.mesh.field.add("Threshold", 4)
    gmsh.model.mesh.field.setNumber(4, "InField", 3)
    gmsh.model.mesh.field.setNumber(4, "SizeMin", lc_outer)
    gmsh.model.mesh.field.setNumber(4, "SizeMax", lc_inner)
    gmsh.model.mesh.field.setNumber(4, "DistMin", 0.0)
    gmsh.model.mesh.field.setNumber(4, "DistMax", annulus_width)

    # Take the minimum of both threshold fields
    gmsh.model.mesh.field.add("Min", 5)
    gmsh.model.mesh.field.setNumbers(5, "FieldsList", [2, 4])
    gmsh.model.mesh.field.setAsBackgroundMesh(5)

    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay
    gmsh.option.setNumber("Mesh.ElementOrder", 1)

    gmsh.model.mesh.generate(2)

    gmsh.fltk.run()

    tmp_msh = output_file.rsplit(".", 1)[0] + ".msh"
    gmsh.write(tmp_msh)
    gmsh.finalize()

    msh_to_xdmf(tmp_msh, output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a 2D triangular mesh.")
    subparsers = parser.add_subparsers(dest="geometry", required=True)

    # --- unit square ---
    sq = subparsers.add_parser("square", help="Unit square [0,1]x[0,1]")
    sq.add_argument("-n", "--segments", type=int, nargs="+", metavar="N",
                    default=[10],
                    help="Number of segments per edge. "
                         "Provide 1 value (uniform) or 4 values (bottom right top left). Default: 10")
    sq.add_argument("-o", "--output", default="unit_square.xdmf", metavar="FILE")

    # --- unit circle ---
    ci = subparsers.add_parser("circle", help="Unit circle")
    ci.add_argument("--lc-boundary", type=float, default=0.1, metavar="H",
                    help="Element size at the boundary. Default: 0.1")
    ci.add_argument("--lc-center", type=float, default=0.1, metavar="H",
                    help="Element size at the center. Default: 0.1")
    ci.add_argument("-o", "--output", default="unit_circle.xdmf", metavar="FILE")

    # --- annulus ---
    an = subparsers.add_parser("annulus", help="Unit circle with concentric circular hole")
    an.add_argument("--hole-radius", type=float, default=0.3, metavar="R",
                    help="Radius of the inner hole (0 < R < 1). Default: 0.3")
    an.add_argument("--lc-outer", type=float, default=0.1, metavar="H",
                    help="Element size at the outer boundary (r=1). Default: 0.1")
    an.add_argument("--lc-inner", type=float, default=0.1, metavar="H",
                    help="Element size at the inner boundary (r=hole_radius). Default: 0.1")
    an.add_argument("-o", "--output", default="annulus.xdmf", metavar="FILE")

    args = parser.parse_args()

    if args.geometry == "square":
        if len(args.segments) == 1:
            segs = args.segments * 4
        elif len(args.segments) == 4:
            segs = args.segments
        else:
            parser.error("--segments expects 1 or 4 values")
        generate_unit_square_mesh(segs, output_file=args.output)

    elif args.geometry == "circle":
        generate_unit_circle_mesh(args.lc_boundary, args.lc_center, output_file=args.output)

    elif args.geometry == "annulus":
        generate_annulus_mesh(args.hole_radius, args.lc_outer, args.lc_inner, output_file=args.output)
