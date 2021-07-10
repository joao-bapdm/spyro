import firedrake
import spyro

# Set parameters
model = {}

# FEM parameters
model["opts"] = {"method": "KMV",
                 "variant": "KMV",
                 "degree": 1,
                 "dimension": 2,
                 "cmin": 1.5,
                 "cmax": 4.5,
                 "rmin": 0.1,
                 "timestepping": "explicit"}
# Mesh parameters
model["mesh"] = {"nz": 349,
                 "nx": 1173,
                 "Lz": 3.0,
                 "Lx": 10.0,
                 "Ly": 0.0}
# PML parameters
model["PML"] = {"status": False,
                "outer_bc": "non-reflective"}
# Parallelism
model["parallelism"] = {"type": "off"}

# Save base model file
spyro.io.save_model(model, "ls_kadu_base.json")

# Create communicator
comm = spyro.utils.mpi_init(model)

# Create mesh
mesh, V = spyro.utils.create_mesh(model, comm, quad=False)

# Load salt model
salt = spyro.utils.load_velocity_model(model, V, source_file="kadu_salt.hdf5")
salt_location = salt.dat.data == 1

# Linear profile
z, x = firedrake.SpatialCoordinate(mesh)
profile = 1.5 + (4 - 1.5) / 3 * z

# Ensemble salt velocity model
vp_salt = firedrake.interpolate(profile, V)
vp_salt.dat.data[salt_location] = 4.5

firedrake.File("vp_salt.pvd").write(vp_salt)

# Save velocity model
spyro.utils.save_velocity_model(comm, vp_salt, "vp_ls_kadu_complete.hdf5")

