from mpi4py import MPI
from firedrake import *

import spyro
import SeismicMesh

model = {}
model["opts"] = {
    "method": "KMV",
    "degree": 1,  # p order
    "dimension": 2,  # dimension
    "quadrature": "KMV",
}
model["mesh"] = {
    "Lz": 1.50,  # depth in km - always positive
    "Lx": 1.50,  # width in km - always positive
    "Ly": 0.0,  # thickness in km - always positive
    "meshfile": "meshes/immersed_disk_guess_vp.msh",
    "initmodel": "velocity_models/immersed_disk_guess_vp.hdf5",
    "truemodel": "velocity_models/immersed_disk_true_vp.hdf5",
}
model["PML"] = {
    "status": True,  # true,  # true or false
    "outer_bc": "non-reflective",  #  dirichlet, neumann, non-reflective (outer boundary condition)
    "damping_type": "polynomial",  # polynomial, hyperbolic, shifted_hyperbolic
    "exponent": 2,
    "cmax": 5.0,  # maximum acoustic wave velocity in pml - km/s
    "R": 0.001,  # theoretical reflection coefficient
    "lz": 0.50,  # thickness of the pml in the z-direction (km) - always positive
    "lx": 0.50,  # thickness of the pml in the x-direction (km) - always positive
    "ly": 0.0,  # thickness of the pml in the y-direction (km) - always positive
}
recvs = spyro.create_transect((-0.10, 0.30), (-0.10, 1.20), 200)
sources = spyro.create_transect((-0.05, 0.30), (-0.10, 1.20), 4)
model["acquisition"] = {
    "source_type": "Ricker",
    "num_sources": len(sources),
    "source_pos": sources,
    "frequency": 5.0,
    "delay": 1.0,
    "amplitude": 1.0,
    "num_receivers": len(recvs),
    "receiver_locations": recvs,
}
model["timeaxis"] = {
    "t0": 0.0,  #  initial time for event
    "tf": 2.0,  # final time for event
    "dt": 0.001,  # timestep size
    "nspool": 100,  # how frequently to output solution to pvds
    "fspool": 10,  # how frequently to save solution to ram
    "skip": 2,
}
model["parallelism"] = {
    "type": "automatic",  # options: automatic (same number of cores for evey processor), custom, off
    "custom_cores_per_shot": [],  # only if the user wants a different number of cores for every shot.
    # input is a list of integers with the length of the number of shots.
}
#### end of options ####


def calculate_indicator_from_vp(vp):
    """Create an indicator function
    assumes the sudomains are labeled 10 and 11
    """
    dgV = FunctionSpace(mesh, "DG", 0)
    cond = conditional(vp > 4.4, -1, 1)
    indicator = Function(dgV, name="indicator").interpolate(cond)
    return indicator


def update_velocity(q, vp):
    """Update the velocity (material properties)
    based on the indicator function
    """
    sd1 = SubDomainData(q < 0)
    sd2 = SubDomainData(q > 0)

    vp.interpolate(Constant(4.5), subset=sd1)
    vp.interpolate(Constant(2.0), subset=sd2)

    evolution_of_velocity.write(vp, name="control")
    return vp


def create_weighting_function(V):
    """Create a weighting function to mask the gradient"""
    # a weighting function that produces large values near the boundary
    # to diminish the gradient calculation near the boundary of the domain
    m = V.ufl_domain()
    W2 = VectorFunctionSpace(m, V.ufl_element())
    coords = interpolate(m.coordinates, W2)
    z, x = coords.dat.data[:, 0], coords.dat.data[:, 1]

    # a weighting function that produces large values near the boundary
    # to diminish the gradient calculation near the boundary of the domain
    disk0 = SeismicMesh.Disk([-0.75, 0.75], 0.40)
    pts = np.column_stack((z[:, None], x[:, None]))
    d = disk0.eval(pts)
    d[d < 0] = 0.0
    vals = 1 + 1000.0 * d
    wei = Function(V, vals, name="weighting_function")
    File("weighting_function.pvd").write(wei)
    return wei


def calculate_functional(model, mesh, comm, vp, sources, receivers):
    """Calculate the l2-norm functional"""
    print("Computing the functional", flush=True)
    J_local = np.zeros((1))
    J_total = np.zeros((1))
    for sn in range(model["acquisition"]["num_sources"]):
        if spyro.io.is_owner(comm, sn):
            guess, guess_dt, guess_recv = spyro.solvers.Leapfrog_level_set(
                model, mesh, comm, vp, sources, receivers, source_num=sn
            )
            p_exact_recv = spyro.io.load_shots(
                "shots/forward_exact_level_set" + str(sn) + ".dat"
            )
            residual = spyro.utils.evaluate_misfit(
                model,
                comm,
                guess_recv,
                p_exact_recv,
            )
            J_local[0] += spyro.utils.compute_functional(model, comm, residual)
    if comm.ensemble_comm.size > 1:
        COMM_WORLD.Allreduce(J_local, J_total, op=MPI.SUM)
        J_total[0] /= comm.ensemble_comm.size
    return (
        J_total[0],
        guess,
        guess_dt,
        residual,
    )


def calculate_gradient(model, mesh, comm, vp, guess, guess_dt, weighting, residual):
    """Calculate the shape gradient"""
    print("Computing the gradient", flush=True)
    VF = VectorFunctionSpace(mesh, model["opts"]["method"], model["opts"]["degree"])
    theta = Function(VF, name="gradient")
    for sn in range(model["acquisition"]["num_sources"]):
        if spyro.io.is_owner(comm, sn):
            theta_local = spyro.solvers.Leapfrog_adjoint_level_set(
                model,
                mesh,
                comm,
                vp,
                guess,
                guess_dt,
                weighting,
                residual,
                source_num=sn,
                output=True,
            )
    # sum shape gradient if ensemble parallelism here
    if comm.ensemble_comm.size > 1:
        comm.ensemble_comm.Allreduce(
            theta_local.dat.data[:], theta.dat.data[:], op=MPI.SUM
        )
    else:
        theta = theta_local
    return theta


def model_update(mesh, indicator, theta, step):
    """Solve a transport equation to move the subdomains around based
    on the shape gradient which hopefully minimizes the functional.
    """
    print("Updating the shape...", flush=True)
    indicator_new = spyro.solvers.advect(
        mesh, indicator, step * theta, number_of_timesteps=100
    )
    return indicator_new


def optimization(model, mesh, V, comm, vp, sources, receivers, max_iter=10):
    """Optimization with a line search algorithm"""
    beta0 = beta0_init = 1.5
    max_ls = 3
    gamma = gamma2 = 0.8

    indicator = calculate_indicator_from_vp(vp)

    # the file that contains the shape gradient each iteration
    grad_file = File("theta.pvd")

    weighting = create_weighting_function(V)

    ls_iter = 0
    iter_num = 0
    # some very large number to start for the functional
    J_old = 9999999.0
    while iter_num < max_iter:
        # calculate the new functional for the new model
        J_new, guess_new, guess_dt_new, residual_new = calculate_functional(
            model, mesh, comm, vp, sources, receivers
        )
        # compute the shape gradient for the new domain
        theta = calculate_gradient(
            model, mesh, comm, vp, guess_new, guess_dt_new, weighting, residual_new
        )
        grad_file.write(theta, name="gradient")
        # update the new shape...solve transport equation
        indicator_new = model_update(mesh, indicator, theta, beta0)
        # update the velocity
        vp_new = update_velocity(indicator_new, vp)
        # using some basic logic attempt to reduce the functional
        print(J_new, flush=True)
        if J_new < J_old:
            print(
                "Iteration "
                + str(iter_num)
                + " : Accepting shape update...functional is: "
                + str(J_new),
                flush=True,
            )
            iter_num += 1
            # accept new domain
            J_old = J_new
            guess = guess_new
            guess_dt = guess_dt_new
            residual = residual_new
            indicator = indicator_new
            vp = vp_new
            # update step
            if ls_iter == max_ls:
                beta0 = max(beta0 * gamma2, 0.1 * beta_0_init)
            elif ls_iter == 0:
                beta0 = min(beta0 / gamma2, 1.0)
            else:
                # no change to step
                beta0 = beta0
            ls_iter = 0
        elif ls_iter < 3:
            print("Line search " + str(ls_iter) + "...reducing step...", flush=True)
            # advance the line search counter
            ls_iter += 1
            # reduce step length by gamma
            beta0 *= gamma
            # now solve the transport equation over again
            # but with the reduced step
        else:
            raise ValueError("Failed to reduce the functional...")

    return vp


# run the script

# visualize the updates
evolution_of_velocity = File("evolution_of_velocity.pvd")

comm = spyro.utils.mpi_init(model)

mesh, V = spyro.io.read_mesh(model, comm)

vp = spyro.io.interpolate(model, mesh, V, guess=True)

# The guess velocity model
File("guess.pvd").write(vp)

q = calculate_indicator_from_vp(vp)

sources = spyro.Sources(model, mesh, V, comm).create()

receivers = spyro.Receivers(model, mesh, V, comm).create()

# run the optimization based on a line search for max_iter iterations
vp = optimization(model, mesh, V, comm, vp, sources, receivers, max_iter=5)