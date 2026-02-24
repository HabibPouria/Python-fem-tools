
"""
Thermo-Mechanical Hotspot (Professional, Class-Based FEM)
========================================================
What this script demonstrates (industry-style):
- Clean architecture using classes (Mesh, Material, Solvers, BCs)
- Q4 bilinear FEM with 2x2 Gauss integration
- Steady heat conduction with Gaussian internal heat source
- Thermoelastic plane-stress mechanics driven by thermal eigenstrain
- Boundary-condition Strategy pattern (AllFixedBC, SemiFixedBC)
- Post-processing: Temperature, displacement magnitude, von Mises stress
- Saves website-ready PNGs

Outputs:
- thermal_T.png
- thermo_disp.png
- thermo_vonmises.png

Dependencies:
- numpy, matplotlib, scipy
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Protocol, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.tri as mtri

# =============================================================================
# Utilities: Q4 shape functions, 2x2 Gauss, Jacobian, gradients
# =============================================================================
def shape_Q4(xi: float, eta: float) -> np.ndarray:
    # Node order: 0(-1,-1), 1(+1,-1), 2(+1,+1), 3(-1,+1)
    return 0.25 * np.array(
        [(1 - xi) * (1 - eta),
         (1 + xi) * (1 - eta),
         (1 + xi) * (1 + eta),
         (1 - xi) * (1 + eta)],
        dtype=float
    )


def dN_dxi_eta_Q4(xi: float, eta: float) -> Tuple[np.ndarray, np.ndarray]:
    dN_dxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)], dtype=float)
    dN_deta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)], dtype=float)
    return dN_dxi, dN_deta


def gauss_2x2() -> Tuple[list[Tuple[float, float]], list[float]]:
    gp = 1.0 / np.sqrt(3.0)
    pts = [(-gp, -gp), (gp, -gp), (gp, gp), (-gp, gp)]
    wts = [1.0, 1.0, 1.0, 1.0]
    return pts, wts


def jacobian_and_grads(coords_elem: np.ndarray, xi: float, eta: float) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    coords_elem: (4,2)
    returns detJ, dNdx(4,), dNdy(4,)
    """
    dN_dxi, dN_deta = dN_dxi_eta_Q4(xi, eta)
    x = coords_elem[:, 0]
    y = coords_elem[:, 1]

    J = np.array([
        [np.dot(dN_dxi, x), np.dot(dN_deta, x)],
        [np.dot(dN_dxi, y), np.dot(dN_deta, y)],
    ], dtype=float)

    detJ = float(np.linalg.det(J))
    if detJ <= 0.0:
        raise ValueError(f"Non-positive detJ={detJ}. Check mesh or element ordering.")

    invJ = np.linalg.inv(J)
    grads = invJ @ np.vstack((dN_dxi, dN_deta))  # (2,4)
    dNdx = grads[0, :]
    dNdy = grads[1, :]
    return detJ, dNdx, dNdy


# =============================================================================
# Core classes: Mesh, Material
# =============================================================================
@dataclass(frozen=True)
class Mesh:
    Lx: float
    Ly: float
    nx: int
    ny: int

    coords: np.ndarray
    conn: np.ndarray
    X: np.ndarray
    Y: np.ndarray

    @staticmethod
    def structured_Q4(Lx: float, Ly: float, nx: int, ny: int) -> "Mesh":
        x = np.linspace(0.0, Lx, nx + 1)
        y = np.linspace(0.0, Ly, ny + 1)
        X, Y = np.meshgrid(x, y, indexing="xy")
        coords = np.column_stack([X.ravel(), Y.ravel()])

        def node_id(i: int, j: int) -> int:
            return j * (nx + 1) + i

        conn = []
        for j in range(ny):
            for i in range(nx):
                n0 = node_id(i, j)
                n1 = node_id(i + 1, j)
                n2 = node_id(i + 1, j + 1)
                n3 = node_id(i, j + 1)
                conn.append([n0, n1, n2, n3])

        return Mesh(Lx=Lx, Ly=Ly, nx=nx, ny=ny,
                    coords=coords, conn=np.array(conn, dtype=int), X=X, Y=Y)

    @property
    def nnode(self) -> int:
        return self.coords.shape[0]

    @property
    def nelem(self) -> int:
        return self.conn.shape[0]

    def boundary_nodes(self, tol: float = 1e-12) -> np.ndarray:
        c = self.coords
        b = np.where(
            (np.abs(c[:, 0] - 0.0) < tol) |
            (np.abs(c[:, 0] - self.Lx) < tol) |
            (np.abs(c[:, 1] - 0.0) < tol) |
            (np.abs(c[:, 1] - self.Ly) < tol)
        )[0]
        return b


@dataclass(frozen=True)
class Material:
    name: str
    E: float
    nu: float
    alpha: float  # thermal expansion [1/K]

    def C_plane_stress(self) -> np.ndarray:
        fac = self.E / (1.0 - self.nu**2)
        return fac * np.array([
            [1.0, self.nu, 0.0],
            [self.nu, 1.0, 0.0],
            [0.0, 0.0, (1.0 - self.nu) / 2.0],
        ], dtype=float)


# =============================================================================
# Boundary Conditions: Strategy Pattern
# =============================================================================
class BoundaryCondition(Protocol):
    def apply_scalar(self, A: lil_matrix, b: np.ndarray, mesh: Mesh) -> Tuple[lil_matrix, np.ndarray]:
        ...

    def apply_vector(self, K: lil_matrix, f: np.ndarray, mesh: Mesh) -> Tuple[lil_matrix, np.ndarray]:
        ...


@dataclass(frozen=True)
class AllDirichletThermalBC:
    """Thermal: T = T_amb on all boundaries."""
    T_amb: float

    def apply_scalar(self, A: lil_matrix, b: np.ndarray, mesh: Mesh) -> Tuple[lil_matrix, np.ndarray]:
        nodes = mesh.boundary_nodes()
        for n in nodes:
            A[n, :] = 0.0
            A[:, n] = 0.0
            A[n, n] = 1.0
            b[n] = self.T_amb
        return A, b


@dataclass(frozen=True)
class AllFixedBC:
    """Mechanics: u=v=0 on all boundaries (strong constraint -> high stress)."""
    def apply_vector(self, K: lil_matrix, f: np.ndarray, mesh: Mesh) -> Tuple[lil_matrix, np.ndarray]:
        nodes = mesh.boundary_nodes()
        for n in nodes:
            for dof, val in [(2*n, 0.0), (2*n + 1, 0.0)]:
                K[dof, :] = 0.0
                K[:, dof] = 0.0
                K[dof, dof] = 1.0
                f[dof] = val
        return K, f


@dataclass(frozen=True)
class SemiFixedBC:
    """
    Mechanics: more realistic packaging-style anchor:
    - u=0 on left edge
    - v=0 on bottom edge
    - anchor corner (0,0): u=v=0 (remove rigid motion cleanly)
    """
    def apply_vector(self, K: lil_matrix, f: np.ndarray, mesh: Mesh) -> Tuple[lil_matrix, np.ndarray]:
        c = mesh.coords
        tol = 1e-12
        left = np.where(np.abs(c[:, 0] - 0.0) < tol)[0]
        bottom = np.where(np.abs(c[:, 1] - 0.0) < tol)[0]
        corner = np.where((np.abs(c[:, 0]) < tol) & (np.abs(c[:, 1]) < tol))[0][0]

        fixed = set()
        for n in left:
            fixed.add((2*n, 0.0))        # u=0
        for n in bottom:
            fixed.add((2*n + 1, 0.0))    # v=0
        fixed.add((2*corner, 0.0))
        fixed.add((2*corner + 1, 0.0))

        for dof, val in fixed:
            K[dof, :] = 0.0
            K[:, dof] = 0.0
            K[dof, dof] = 1.0
            f[dof] = val

        return K, f


# =============================================================================
# Physics: Heat source model
# =============================================================================
@dataclass(frozen=True)
class GaussianHotspot:
    Q0: float
    x0: float
    y0: float
    a: float

    def q(self, x: float, y: float) -> float:
        r2 = (x - self.x0)**2 + (y - self.y0)**2
        return float(self.Q0 * np.exp(-r2 / (self.a**2)))


# =============================================================================
# Solvers
# =============================================================================
@dataclass
class ThermalSolver:
    mesh: Mesh
    k_th: float
    bc: AllDirichletThermalBC
    source: GaussianHotspot

    def assemble(self) -> Tuple[lil_matrix, np.ndarray]:
        n = self.mesh.nnode
        K = lil_matrix((n, n), dtype=float)
        F = np.zeros(n, dtype=float)

        pts, wts = gauss_2x2()

        for nodes in self.mesh.conn:
            coords_elem = self.mesh.coords[nodes, :]
            Ke = np.zeros((4, 4), dtype=float)
            Fe = np.zeros(4, dtype=float)

            for (xi, eta), w in zip(pts, wts):
                N = shape_Q4(xi, eta)
                detJ, dNdx, dNdy = jacobian_and_grads(coords_elem, xi, eta)

                B = np.vstack((dNdx, dNdy))  # (2,4)
                Ke += (B.T @ (self.k_th * np.eye(2)) @ B) * detJ * w

                x_gp = float(np.dot(N, coords_elem[:, 0]))
                y_gp = float(np.dot(N, coords_elem[:, 1]))
                qgp = self.source.q(x_gp, y_gp)
                Fe += N * qgp * detJ * w

            for a in range(4):
                A = nodes[a]
                F[A] += Fe[a]
                for b in range(4):
                    Bn = nodes[b]
                    K[A, Bn] += Ke[a, b]

        return K, F

    def solve(self) -> np.ndarray:
        K, F = self.assemble()
        K, F = self.bc.apply_scalar(K, F, self.mesh)
        T = spsolve(csr_matrix(K), F)
        return np.asarray(T, dtype=float)


@dataclass
class ThermoElasticSolver:
    mesh: Mesh
    material: Material
    T: np.ndarray
    T_ref: float
    bc: BoundaryCondition  # any BC implementing apply_vector

    def assemble(self) -> Tuple[lil_matrix, np.ndarray]:
        nnode = self.mesh.nnode
        ndof = 2 * nnode
        K = lil_matrix((ndof, ndof), dtype=float)
        f = np.zeros(ndof, dtype=float)

        C = self.material.C_plane_stress()
        pts, wts = gauss_2x2()

        for nodes in self.mesh.conn:
            coords_elem = self.mesh.coords[nodes, :]
            Te = self.T[nodes]

            Ke = np.zeros((8, 8), dtype=float)
            fe = np.zeros(8, dtype=float)

            for (xi, eta), w in zip(pts, wts):
                N = shape_Q4(xi, eta)
                detJ, dNdx, dNdy = jacobian_and_grads(coords_elem, xi, eta)

                # B (3x8)
                B = np.zeros((3, 8), dtype=float)
                for a in range(4):
                    B[0, 2*a]     = dNdx[a]
                    B[1, 2*a + 1] = dNdy[a]
                    B[2, 2*a]     = dNdy[a]
                    B[2, 2*a + 1] = dNdx[a]

                Ke += (B.T @ C @ B) * detJ * w

                Tgp = float(np.dot(N, Te))
                dT = Tgp - self.T_ref
                eps_th = self.material.alpha * dT * np.array([1.0, 1.0, 0.0], dtype=float)

                # Equivalent thermal load: ∫ B^T C eps_th dΩ
                fe += (B.T @ (C @ eps_th)) * detJ * w

            # Global dofs
            dofs = []
            for a in range(4):
                dofs.extend([2*nodes[a], 2*nodes[a] + 1])

            for i_local, I in enumerate(dofs):
                f[I] += fe[i_local]
                for j_local, J in enumerate(dofs):
                    K[I, J] += Ke[i_local, j_local]

        return K, f

    def solve(self) -> np.ndarray:
        K, f = self.assemble()
        K, f = self.bc.apply_vector(K, f, self.mesh)  # type: ignore[attr-defined]
        U = spsolve(csr_matrix(K), f)
        return np.asarray(U, dtype=float)


# =============================================================================
# Post-processing
# =============================================================================
class PostProcessor:
    @staticmethod
    def displacement_magnitude(mesh: Mesh, U: np.ndarray) -> np.ndarray:
        ux = U[0::2]
        uy = U[1::2]
        return np.sqrt(ux**2 + uy**2)

    @staticmethod
    def von_mises_nodes(mesh: Mesh, material: Material, U: np.ndarray, T: np.ndarray, T_ref: float) -> np.ndarray:
        """
        N-weighted averaging of Gauss-point stresses to nodes, then plane-stress von Mises:
            vm = sqrt(sxx^2 - sxx*syy + syy^2 + 3*sxy^2)
        """
        nnode = mesh.nnode
        sig_node = np.zeros((nnode, 3), dtype=float)
        counts = np.zeros(nnode, dtype=float)

        C = material.C_plane_stress()
        pts, wts = gauss_2x2()

        for nodes in mesh.conn:
            coords_elem = mesh.coords[nodes, :]
            Te = T[nodes]

            ue = np.zeros(8, dtype=float)
            for a in range(4):
                ue[2*a] = U[2*nodes[a]]
                ue[2*a + 1] = U[2*nodes[a] + 1]

            for (xi, eta), w in zip(pts, wts):
                N = shape_Q4(xi, eta)
                detJ, dNdx, dNdy = jacobian_and_grads(coords_elem, xi, eta)

                B = np.zeros((3, 8), dtype=float)
                for a in range(4):
                    B[0, 2*a]     = dNdx[a]
                    B[1, 2*a + 1] = dNdy[a]
                    B[2, 2*a]     = dNdy[a]
                    B[2, 2*a + 1] = dNdx[a]

                eps = B @ ue
                Tgp = float(np.dot(N, Te))
                dT = Tgp - T_ref
                eps_th = material.alpha * dT * np.array([1.0, 1.0, 0.0], dtype=float)

                sig = C @ (eps - eps_th)  # [sxx, syy, sxy]

                for a in range(4):
                    A = nodes[a]
                    sig_node[A, :] += N[a] * sig
                    counts[A] += N[a]

        sig_node /= counts[:, None]
        sxx, syy, sxy = sig_node[:, 0], sig_node[:, 1], sig_node[:, 2]
        vm = np.sqrt(sxx**2 - sxx*syy + syy**2 + 3.0*sxy**2)
        return vm
# =====================================================
#  BC Visualization Function  
# =====================================================
def plot_boundary_conditions(mesh: Mesh, bc, filename="bc_visualization.png"):
    """
    Visualize mechanical boundary conditions.
    Red arrows  -> u fixed
    Blue arrows -> v fixed
    """
    coords = mesh.coords
    plt.figure()

    # Plot mesh lines
    for j in range(mesh.ny + 1):
        plt.plot(mesh.X[j, :], mesh.Y[j, :], linewidth=0.5, color='gray')
    for i in range(mesh.nx + 1):
        plt.plot(mesh.X[:, i], mesh.Y[:, i], linewidth=0.5, color='gray')

    # Determine constrained DOFs
    fixed_u = []
    fixed_v = []

    if isinstance(bc, AllFixedBC):
        nodes = mesh.boundary_nodes()
        fixed_u = nodes
        fixed_v = nodes

    elif isinstance(bc, SemiFixedBC):
        tol = 1e-12
        left = np.where(np.abs(coords[:, 0]) < tol)[0]
        bottom = np.where(np.abs(coords[:, 1]) < tol)[0]

        fixed_u = left
        fixed_v = bottom

    # Plot u-fixed (horizontal arrows)
    if len(fixed_u) > 0:
        plt.quiver(
            coords[fixed_u, 0],
            coords[fixed_u, 1],
            -0.02, 0,
            angles='xy',
            scale_units='xy',
            scale=1,
            color='red',
            label='u fixed'
        )

    # Plot v-fixed (vertical arrows)
    if len(fixed_v) > 0:
        plt.quiver(
            coords[fixed_v, 0],
            coords[fixed_v, 1],
            0, -0.02,
            angles='xy',
            scale_units='xy',
            scale=1,
            color='blue',
            label='v fixed'
        )

    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(f"Mechanical Boundary Conditions ({bc.__class__.__name__})")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.show()

# =====================================================
#  Draw original mesh (light gray)
#  Draw deformed mesh (scaled)
#  Draw displacement vectors as arrows 
# =====================================================


def plot_deformation_with_arrows(mesh: Mesh, U: np.ndarray,
                                 scale=50.0,
                                 arrow_stride=3,
                                 filename="deformation_arrows.png"):
    """
    Plot deformed mesh + displacement arrows.
    scale        : visual scaling factor
    arrow_stride : skip nodes for clarity (1 = all nodes)
    """

    ux = U[0::2]
    uy = U[1::2]

    Uxg = ux.reshape((mesh.ny + 1, mesh.nx + 1))
    Uyg = uy.reshape((mesh.ny + 1, mesh.nx + 1))

    # Deformed coordinates
    Xd = mesh.X + scale * Uxg
    Yd = mesh.Y + scale * Uyg

    plt.figure(figsize=(8,6))

    # --- Original mesh ---
    for j in range(mesh.ny + 1):
        plt.plot(mesh.X[j, :], mesh.Y[j, :],
                 color='lightgray', linewidth=0.6)
    for i in range(mesh.nx + 1):
        plt.plot(mesh.X[:, i], mesh.Y[:, i],
                 color='lightgray', linewidth=0.6)

    # --- Deformed mesh ---
    for j in range(mesh.ny + 1):
        plt.plot(Xd[j, :], Yd[j, :],
                 color='black', linewidth=0.8)
    for i in range(mesh.nx + 1):
        plt.plot(Xd[:, i], Yd[:, i],
                 color='black', linewidth=0.8)

    # --- Displacement arrows ---
    plt.quiver(
        mesh.X[::arrow_stride, ::arrow_stride],
        mesh.Y[::arrow_stride, ::arrow_stride],
        Uxg[::arrow_stride, ::arrow_stride] * scale,
        Uyg[::arrow_stride, ::arrow_stride] * scale,
        color='red',
        angles='xy',
        scale_units='xy',
        scale=1
    )

    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("Deformed Mesh with Displacement Vectors")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.show()
# =============================================================================
# plot_scalar_on_deformed_mesh
# =============================================================================
def plot_scalar_on_deformed_mesh(mesh: Mesh,
                                 U: np.ndarray,
                                 scalar: np.ndarray,
                                 title: str,
                                 cbar_label: str,
                                 scale: float = 80.0,
                                 filename: str = "deformed_contour.png",
                                 nlevels: int = 30):
    """
    Plot nodal scalar values on the DEFORMED configuration.

    - Builds a triangulation by splitting each Q4 element into 2 triangles.
    - Uses deformed node coordinates: x' = x + scale*ux, y' = y + scale*uy
    """

    # --- deformed nodal coordinates ---
    x = mesh.coords[:, 0].copy()
    y = mesh.coords[:, 1].copy()
    ux = U[0::2]
    uy = U[1::2]
    xd = x + scale * ux
    yd = y + scale * uy

    # --- build triangles from Q4 connectivity ---
    # Q4 nodes: [n0, n1, n2, n3]
    # split into (n0,n1,n2) and (n0,n2,n3)
    tris = np.zeros((mesh.nelem * 2, 3), dtype=int)
    for e, nodes in enumerate(mesh.conn):
        n0, n1, n2, n3 = nodes
        tris[2*e + 0, :] = [n0, n1, n2]
        tris[2*e + 1, :] = [n0, n2, n3]

    tri = mtri.Triangulation(xd, yd, tris)

    plt.figure(figsize=(8, 5.5))
    cf = plt.tricontourf(tri, scalar, levels=nlevels)
    plt.colorbar(cf, label=cbar_label)

    # optional: draw deformed mesh edges (light)
    plt.triplot(tri, linewidth=0.25, alpha=0.35)

    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(title)
    plt.xlabel("x (deformed)")
    plt.ylabel("y (deformed)")
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.show()


# =============================================================================
# Main demo (Hotspot thermoelastic)
# =============================================================================
def main():
    # -------------------------
    # Settings (edit freely)
    # -------------------------
    mesh = Mesh.structured_Q4(Lx=1.0, Ly=1.0, nx=40, ny=40)

    # Thermal
    k_th = 20.0
    T_amb = 293.15
    T_ref = 293.15
    source = GaussianHotspot(Q0=2.0e6, x0=0.5, y0=0.5, a=0.08)
    thermal_bc = AllDirichletThermalBC(T_amb=T_amb)

    # Mechanics
    steel = Material(name="Steel", E=210e9, nu=0.30, alpha=12e-6)

    # Choose one:
    # mech_bc = AllFixedBC()
    mech_bc = SemiFixedBC()
    
    # --- Visualize Boundary Conditions ---
    plot_boundary_conditions(mesh, mech_bc)
    
    # -------------------------
    # Solve thermal
    # -------------------------
    print("=== Thermal solve ===")
    thermal = ThermalSolver(mesh=mesh, k_th=k_th, bc=thermal_bc, source=source)
    T = thermal.solve()
    print(f"T: min={T.min():.3f} K, max={T.max():.3f} K")

    # -------------------------
    # Solve thermoelastic
    # -------------------------
    print("=== Thermoelastic solve ===")
    mech = ThermoElasticSolver(mesh=mesh, material=steel, T=T, T_ref=T_ref, bc=mech_bc)
    U = mech.solve()
    
    plot_deformation_with_arrows(mesh, U, scale=80, arrow_stride=4)
    
    umag = PostProcessor.displacement_magnitude(mesh, U)
    vm = PostProcessor.von_mises_nodes(mesh, steel, U, T, T_ref)

    # Show ALL results on the deformed shape
    plot_scalar_on_deformed_mesh(
        mesh, U, T,
        title="Temperature Field on Deformed Shape",
        cbar_label="Temperature (K)",
        scale=80,
        filename="T_on_deformed.png"
    )
    
    plot_scalar_on_deformed_mesh(
        mesh, U, umag,
        title="Displacement Magnitude on Deformed Shape",
        cbar_label="|u| (m)",
        scale=80,
        filename="U_on_deformed.png"
    )
    
    plot_scalar_on_deformed_mesh(
        mesh, U, vm,
        title="von Mises Stress on Deformed Shape",
        cbar_label="von Mises (Pa)",
        scale=80,
        filename="VM_on_deformed.png"
    )

    print(f"|u| max = {umag.max():.6e} m")
    print(f"von Mises max = {vm.max():.6e} Pa")

    # -------------------------
    # Plot + save (website-ready)
    # -------------------------
    TT = T.reshape((mesh.ny + 1, mesh.nx + 1))
    UM = umag.reshape((mesh.ny + 1, mesh.nx + 1))
    VM = vm.reshape((mesh.ny + 1, mesh.nx + 1))

    plt.figure()
    cf = plt.contourf(mesh.X, mesh.Y, TT, levels=30)
    plt.colorbar(cf, label="Temperature (K)")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Steady-State Temperature Field with Gaussian Hotspot")
    plt.tight_layout()
    plt.savefig("thermal_T.png", dpi=200)

    plt.figure()
    cf = plt.contourf(mesh.X, mesh.Y, UM, levels=30)
    plt.colorbar(cf, label="|u| (m)")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Displacement Magnitude (BC={mech_bc.__class__.__name__})")
    plt.tight_layout()
    plt.savefig("thermo_disp.png", dpi=200)

    plt.figure()
    cf = plt.contourf(mesh.X, mesh.Y, VM, levels=30)
    plt.colorbar(cf, label="von Mises (Pa)")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Thermally Induced von Mises Stress")
    plt.tight_layout()
    plt.savefig("thermo_vonmises.png", dpi=200)

    plt.show()
    print("Saved: thermal_T.png, thermo_disp.png, thermo_vonmises.png")


if __name__ == "__main__":
    main()
