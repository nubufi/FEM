import numpy as np
import pandas as pd
from objects import Node


class Element:
    def __init__(self, node1, node2, node3, E, v, t):
        self.node1 = node1
        self.node2 = node2
        self.node3 = node3
        self.E = E
        self.v = v
        self.t = t

    def get_freedoms(self):
        self.freedoms = np.array(
            [
                self.node1.x_label,
                self.node1.y_label,
                self.node2.x_label,
                self.node2.y_label,
                self.node3.x_label,
                self.node3.y_label,
            ]
        )

    def B_matrix(self):
        c1 = (self.node1.x, self.node1.y)
        c2 = (self.node2.x, self.node2.y)
        c3 = (self.node3.x, self.node3.y)
        B_i = c2[1] - c3[1]
        B_j = c3[1] - c1[1]
        B_m = c1[1] - c2[1]
        g_i = c3[0] - c2[0]
        g_j = c1[0] - c3[0]
        g_m = c2[0] - c1[0]
        self.A = (c1[0] * B_i + c2[0] * B_j + c3[0] * B_m) * 0.5

        self.B = (1 / (2 * self.A)) * np.array(
            [
                [B_i, 0, B_j, 0, B_m, 0],
                [0, g_i, 0, g_j, 0, g_m],
                [g_i, B_i, g_j, B_j, g_m, B_m],
            ]
        )

    def D_matrix(self):
        constant = self.E / (1 - self.v ** 2)
        self.D = (
            np.array([[1, self.v, 0], [self.v, 1, 0], [0, 0, (1 - self.v) / 2]])
            * constant
        )

    def k_matrix(self):
        self.B_matrix()
        self.D_matrix()
        self.get_freedoms()
        B_ = np.transpose(self.B)
        self.k = self.t * self.A * np.matmul(B_, np.matmul(self.D, self.B))

    def stress_matrix(self, displacements):
        self.d = np.array([[displacements[i] for i in self.freedoms]]).T
        self.stresses = np.matmul(self.D, np.matmul(self.B, self.d))
        return self.stresses


class FEM:
    def __init__(self, excel_file, F, boundary) -> None:
        self.excel_file = excel_file
        self.K = np.zeros((len(F), len(F)))
        self.F = np.array([F])
        self.boundary = np.array(boundary)
        self.free_nodes = np.where(self.boundary != 0)[0]
        self.all_nodes = np.arange(len(self.boundary))

    def create_node(self, excel, elem_index, node_index):
        x = excel.iloc[elem_index, node_index + 4]
        y = excel.iloc[elem_index, node_index + 5]
        x_label = excel.iloc[elem_index, node_index + 10] - 1
        y_label = excel.iloc[elem_index, node_index + 11] - 1
        D_x = self.boundary[x_label]
        D_y = self.boundary[y_label]
        F_x = self.F[0][x_label]
        F_y = self.F[0][y_label]

        return Node(x, y, x_label, y_label, D_x, D_y, F_x, F_y)

    def create_element(self, excel, elem_index):
        node1 = self.create_node(excel, elem_index, 0)
        node2 = self.create_node(excel, elem_index, 2)
        node3 = self.create_node(excel, elem_index, 4)
        E = excel.iloc[elem_index, 1]
        v = excel.iloc[elem_index, 2]
        t = excel.iloc[elem_index, 3]
        elem = Element(node1, node2, node3, E, v, t)
        elem.k_matrix()
        self.add2global(elem)
        return elem

    def create_data(self):
        excel = pd.read_excel(self.excel_file, skiprows=1)
        self.elements = {}
        for i in range(len(excel)):
            self.elements[f"Elem-{i+1}"] = self.create_element(excel, i)

    def add2global(self, elem):
        for i in elem.freedoms:
            local_i = list(elem.freedoms).index(i)
            for j in elem.freedoms:
                local_j = list(elem.freedoms).index(j)
                self.K[i, j] += elem.k[local_i, local_j]

    def K_reduction(self):
        KR = np.zeros((len(self.free_nodes), len(self.free_nodes)))
        for i in range(len(self.free_nodes)):
            for j in range(len(self.free_nodes)):
                KR[i, j] = self.K[self.free_nodes[i], self.free_nodes[j]]
        return KR

    def F_reduction(self):
        FR = np.zeros((len(self.free_nodes), 1))
        for i in range(len(self.free_nodes)):
            FR[i] = self.F[0][self.free_nodes[i]]
        return FR

    def D(self):
        KR = self.K_reduction()
        KR_inv = np.linalg.inv(KR)
        FR = self.F_reduction()
        DR = np.matmul(KR_inv, FR)
        self.D = np.zeros_like(self.boundary, dtype=float)
        for i in range(len(self.D)):
            if i in self.free_nodes:
                local_index = list(self.free_nodes).index(i)
                self.D[i] = DR[local_index]

    def G(self):
        for elem in self.elements.values():
            elem.stress_matrix(self.D)

    def print_matrix(self, nodes, data, precision=3):
        default_line = "\n" + len(nodes) * "---------+" + "\n"
        text = default_line
        for i in nodes:
            text += "{:^9}|".format(i)
        text += default_line

        for j in range(data.shape[0]):
            for i in range(data.shape[1]):
                text += "{:^9}|".format(np.round(data[j, i], precision))
            text += "\n"

        text += default_line

        print(text)

    def printer(self):
        for i, elem in enumerate(self.elements.values()):
            print(40 * "-" + f" Element {i+1} " + 40 * "-")
            print("k = 1e6 * ")
            self.print_matrix(elem.freedoms + 1, elem.k / 1e6, 3)
            self.print_matrix(
                ["\u03C3x(kPa)", "\u03C3y(kPa)", "\u03C4xy(kPa)"],
                elem.stresses.reshape(1, 3) / 1000,
                3,
            )
            print("d(mm) = ")
            self.print_matrix(
                elem.freedoms + 1, elem.d.reshape(1, len(elem.freedoms)) * 1000, 5
            )
        print(45 * "-" + f" Global " + 45 * "-")
        print("K = 1e6 * ")
        self.print_matrix(self.all_nodes + 1, self.K / 1e6, 3)
        print("F(kN) = ")
        self.print_matrix(self.all_nodes + 1, self.F / 1000, 3)
        print("D(mm) = ")
        self.print_matrix(
            self.all_nodes + 1, self.D.reshape(1, len(self.all_nodes)) * 1000, 5
        )
        print("R(kN) = ")
        self.print_matrix(
            self.all_nodes + 1, self.R.reshape(1, len(self.all_nodes)) / 1000, 5
        )

    def operator(self):
        self.create_data()
        self.D()
        self.R = np.matmul(self.K, self.D) - self.F
        self.G()
        self.printer()


# FEM("Test.xlsx", [0, 0, 0, 0, 14000, 0, 14000, 0], [0, 0, 0, 0, 1, 1, 1, 1]).operator()
FEM(
    "Kaynaklar\\HW # 7.xlsx",
    [0, 0, 0, 0, 0, 0, 0, 0, 0, -30000],
    [1, 1, 1, 1, 0, 0, 0, 0, 1, 1],
).operator()
