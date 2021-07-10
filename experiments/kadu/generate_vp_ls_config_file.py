import firedrake
import spyro
import numpy as np

# Load base parameters
model = spyro.io.load_model("ls_kadu_base.json")

# Acquisition parameters
srcs = np.arange(0.1, 9.95, 0.2)
srcs = srcs[::4]
srcs = np.delete(srcs, 6)
recs = np.arange(0.05, 10, 0.1)
model["acquisition"] = {"source_type": "Ricker",
                        "amplitude": 10.0,
                        "frequency": 2,
                        "delay": 1.0,
                        "source_pos": [[0, src] for src in srcs],
                        "receiver_locations": [[0.05, rec] for rec in recs]}

# Mesh parameters
# model["mesh"]["nz"] = 42
# model["mesh"]["nx"] = 141
model["mesh"]["nz"] = 60
model["mesh"]["nx"] = 200

# Time parameters
model["timeaxis"] = {"t0": 0.0,
                     "tf": 2.6,
                     "dt": 0.001,
                     "nspool": 5,
                     "fspool": 1}
                        
###############################################################################

# Create communicator
comm = spyro.utils.mpi_init(model)

# Create mesh
mesh, V = spyro.utils.create_mesh(model, comm, quad=False, diagonal="left")

# Load salt model
vp_salt = spyro.utils.load_velocity_model(model, V, source_file="vp_ls_kadu_complete.hdf5")
firedrake.File("vp_salt_left.pvd").write(vp_salt)

# Acquisition
sources, _ = spyro.Geometry(model, mesh, V, comm).create()
source = firedrake.interpolate(sum(sources), V)
firedrake.File("source.pvd").write(source)
# Now switch
model["acquisition"]["source_pos"] = model["acquisition"]["receiver_locations"]
sources, _ = spyro.Geometry(model, mesh, V, comm).create()
receiver = firedrake.interpolate(sum(sources), V)
firedrake.File("receiver.pvd").write(receiver)

# Fix acquisition geometry
model["acquisition"]["source_pos"] = [[0, src] for src in srcs]
model["acquisition"]["receiver_locations"] = [[0.05, rec] for rec in recs]
sources, _ = spyro.Geometry(model, mesh, V, comm).create()

# Save config file
model.pop("output")
model.pop("data")
spyro.io.save_model(model, "ls_kadu.json")

# import IPython; IPython.embed(); exit()gg
spyro.utils.save_velocity_model(comm, vp_salt, "vp_ls_kadu.hdf5")
