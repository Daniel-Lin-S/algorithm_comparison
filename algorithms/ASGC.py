class grid_file:
    def __init__(self, data, grid_num):
        """
        Create a grid file used to store grid blocks

        :param data: 2-d numpy array, each row represents a data point
        :param dim: int, number of dimensions
        :param grid_num: number of grids in each dimension
            if given a integer, it applies to all dimensions
            if a list of int is given, each number corresponds to number of grids in a dimension
        """
        self.data = data
        self.dim = data.shape[1]
        if isinstance(grid_num, int):
            grid_num = [grid_num] * self.dim
        if len(grid_num) != self.dim:
            raise Exception('length of grid_num must be the same as number of dimensions of data')
        
        self.blocks = []  # stores list of blocks