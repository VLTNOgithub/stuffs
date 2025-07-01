def parse_matrix(matrix):
    parsed_matrix = []
    for rgb in matrix:
        if rgb == []:
            rgb = [0, 0, 0]

        parsed_matrix.append(rgb)
    return parsed_matrix