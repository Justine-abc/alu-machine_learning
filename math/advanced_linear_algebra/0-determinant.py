
n that calculates the determinant of a matrix"""


def determinant(matrix):
    """Function that calculates the determinant of a matrix"""
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if len(matrix) > 0:
        for i in matrix:
            if type(i) is not list:
                raise TypeError("matrix must be a list of lists")
    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1
    for i in matrix:
        if len(i) != len(matrix):
            raise ValueError("matrix must be a square matrix")
    if len(matrix) == 1 and len(matrix) == 1:
        return matrix[0][0]
    if len(matrix) == 2:
        return ((matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0]))
    det = []
    for i in range(len(matrix)):
        mini = [[j for j in matrix[i]] for i in range(1, len(matrix))]
        for j in range(len(mini)):
            mini[j].pop(i)
        if i % 2 == 0:
            det.append(matrix[0][i] * determinant(mini))
        if i % 2 == 1:
            det.append(-1 * matrix[0][i] * determinant(mini))
n vi:
# 1. Press 'i' to enter insert mode
# 2. Type/paste your Python code
# 3. Press Esc to exit insert mode
# 4. Type :wq and press Enter to save and quitn vi, save and exit:
# Press Esc, then type :wq and press Enter    return sum(det)
