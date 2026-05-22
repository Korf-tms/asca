import dolfinx as dfx
import scipy
import ufl
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from mpi4py import MPI
from dolfinx.fem.petsc import assemble_matrix


def assemble_laplacian(mesh, mean : float=None, sigma: float=None, seed: int=None):
    """
    Assemble a P(1) FEM laplacian matrix on the given mesh. If mean, sigma,
    and seed are provided the matrix is assembled with a ramdom log-normal
    coefficient field.
    Inputs
    ------
    mesh : file containing mesh in format supported by dolfinx, typically xdmf
    mean : mean of the distribution of the log-normal coefficient field
    sigma : standard deviation of the distribution of the log-normal coefficient field
    seed : random seed for np.random
    Returns
    A_scipy : assembled scipy matrix in CSR format
    kappa : coefficient field, DG(0) function
    """

    rng = np.random.default_rng(seed)
    
    # finite element choice 
    V = dfx.fem.functionspace(mesh, ('Lagrange', 1))
    # auxiliary field for coefficient
    W = dfx.fem.functionspace(mesh, ('DG', 0))
    kappa = dfx.fem.Function(W)
    if mean is not None and sigma is not None:
        kappa.x.array[:] = rng.lognormal(mean=mean, sigma=sigma, size=kappa.x.array.shape)
    else:
        kappa.x.array[:] = 1.0
    
    # variational form and assembly
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = ufl.inner(kappa * ufl.grad(u), ufl.grad(v)) * ufl.dx
    a_form = dfx.fem.form(a)
    A = assemble_matrix(a_form)
    A.assemble()

    # convert to scipy format
    A.convert("seqaij")
    indptr, cols, values = A.getValuesCSR()
    A_scipy = scipy.sparse.csr_matrix((values, cols, indptr), shape=A.size)
    return A_scipy, kappa

def plot_kappa_on_mesh(mesh, kappa, output_file):
    """
    Plot a cell-wise coefficient field on a triangular mesh and save to output_file.
    """

    tdim = mesh.topology.dim
    mesh.topology.create_connectivity(tdim, 0)
    cells_to_vertices = mesh.topology.connectivity(tdim, 0)

    num_cells_local = mesh.topology.index_map(tdim).size_local
    triangles = np.array([cells_to_vertices.links(c) for c in range(num_cells_local)], dtype=np.int32)

    points = mesh.geometry.x[:, :2]
    triangulation = mtri.Triangulation(points[:, 0], points[:, 1], triangles)

    cell_values = np.asarray(kappa.x.array[:num_cells_local])

    plt.figure(figsize=(8, 6))
    trip = plt.tripcolor(triangulation, facecolors=cell_values, edgecolors='k', cmap='viridis')
    plt.colorbar(trip, label='kappa value')
    plt.title('Coefficient Field kappa on Mesh')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        mesh_file_name = sys.argv[1]
    else:
        print("No filename provided, defaulting to expected 'unit_square.xdmf'.")
        mesh_file_name = "unit_square.xdmf"

    with dfx.io.XDMFFile(MPI.COMM_WORLD, mesh_file_name, 'r') as mesh_file:
        mesh = mesh_file.read_mesh(name='Grid')

    # Prepare Laplacian
    A, kappa_values = assemble_laplacian(mesh, mean=0.0, sigma=1e-6)
    scipy.io.mmwrite('fem_laplacian.mtx', A.tocoo())
    print(f"Laplacian written to fem_laplacian.mtx")
    
    plot_kappa_on_mesh(mesh, kappa_values, 'kappa_plot.pdf')
    print(f"kappa plot saved to kappa_plot.pdf")

    # Save DOF coordinates for nice plot of C-F splitting and subgarphs
    V = dfx.fem.functionspace(mesh, ('Lagrange', 1))
    dof_coords = V.tabulate_dof_coordinates()[:, :2]
    np.savetxt('fem_laplacian_coords.csv', dof_coords, delimiter=',', header='x,y', comments='')
    print(f"DOF coordinates written to fem_laplacian_coords.csv ({len(dof_coords)} DOFs)")
