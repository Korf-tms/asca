#read the approximation data
#calculate schurs complement on the original graph
#do shenanigans
'''
def evaluate_approximation():
    # cg needs semi positive definite matrix, even small negatives make issues
    approximation = self.current_approximation + np.eye(self.current_approximation.shape[0]) * 1e-5
    schur = self.current_graph.local_schur_complement() + np.eye(self.current_approximation.shape[0]) * 1e-5
        
    self.file = h5py.File("data/analysis.hdf5", mode="w")
    group = self.file.require_group(f"iteration{self.current_iteration}")
    group.create_dataset(f"asca", data=approximation)
    group.create_dataset(f"schur_complement", data=schur)
    group.create_dataset(f"difference", data=schur - approximation)

    tolerance = 1e-5
    logging.info(f"Matrix symetry check of approximation with tolerance {tolerance}: {np.allclose(approximation, approximation.T, rtol=tolerance, atol=tolerance)}")
    logging.info(f"Matrix symetry check of schur complement with tolerance {tolerance}: {np.allclose(schur, schur.T, rtol=tolerance, atol=tolerance)}")
    sdp_approximation = np.all(np.linalg.eigvalsh(approximation) >= 0)
    logging.info(f"Positive semi-definite check approximation: {sdp_approximation}")
    spd_schur = np.all(np.linalg.eigvalsh(schur) >= 0)
    logging.info(f"Positive semi-definite check schur complement: {spd_schur}")

     if not sdp_approximation or not spd_schur:
         logging.warning("Matrix is not positive semi-definite, cg might not converge.")

    self.cgs_iterations = 0

    b = np.random.rand(approximation.shape[0], 1)
    x, info = cgs(
        A=schur, 
        M=approximation, 
        b=b,
        callback=self.cgs_callback
        )
    logging.info(f"Return value of cgs x: {x}, info: {info}")
    self.file.close()
    logging.info(f"Number of iterations of cgs: {self.cgs_iterations}")

    logging.info(f"Final solution vector: {schur @ x}")
    logging.info(f"Final approximation solution vector: {b}")

def cgs_callback(self, solution_vector):
    group = self.file.require_group(f"iteration{self.current_iteration}")
    group = group.require_group(f"cgs_iterations")
    group.create_dataset(f"cgs_solution_{self.cgs_iterations}", data=solution_vector)
    self.cgs_iterations += 1'''