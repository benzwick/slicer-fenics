import vtk
import qt
import ctk
import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)
import logging

import dolfin


#
# FEniCS_demo_module
#
class FEniCS_demo_module(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "FEniCS demo module"
        self.parent.categories = ["Examples"]
        self.parent.dependencies = []
        # replace with "Firstname Lastname (Organization)"
        self.parent.contributors = ["Ben Zwick (UWA)"]
        self.parent.helpText = """
This is an example of the FEniCS hyperelasticity demo bundled in an extension.
"""
        self.parent.helpText += self.getDefaultModuleDocumentationLink()
        self.parent.acknowledgementText = """
This file was originally developed by Ben Zwick.
"""


#
# FEniCS_demo_moduleWidget
#
class FEniCS_demo_moduleWidget(ScriptedLoadableModuleWidget):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        # Instantiate and connect widgets ...

        #
        # Parameters Area
        #
        parametersCollapsibleButton = ctk.ctkCollapsibleButton()
        parametersCollapsibleButton.text = "Parameters"
        self.layout.addWidget(parametersCollapsibleButton)

        # Layout within the dummy collapsible button
        parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

        #
        # check box to trigger taking screen shots for later use in tutorials
        #
        self.enableScreenshotsFlagCheckBox = qt.QCheckBox()
        self.enableScreenshotsFlagCheckBox.checked = 0
        self.enableScreenshotsFlagCheckBox.setToolTip(
            "If checked, take screen shots for tutorials. Use Save Data to write them to disk.")
        parametersFormLayout.addRow(
            "Enable Screenshots", self.enableScreenshotsFlagCheckBox)

        #
        # input mesh selector
        #
        self.inputSelector = slicer.qMRMLNodeComboBox()
        self.inputSelector.nodeTypes = ["vtkMRMLModelNode"]
        self.inputSelector.selectNodeUponCreation = True
        self.inputSelector.addEnabled = False
        self.inputSelector.removeEnabled = False
        self.inputSelector.noneEnabled = False
        self.inputSelector.showHidden = False
        self.inputSelector.showChildNodeTypes = False
        self.inputSelector.setMRMLScene(slicer.mrmlScene)
        self.inputSelector.setToolTip("Pick input model to use as mesh.")
        parametersFormLayout.addRow(
            "Input Model: ", self.inputSelector)

        #
        # Create Problem Button
        #
        self.createProblemButton = qt.QPushButton("Create Problem")
        self.createProblemButton.toolTip = "Create a new FEniCS problem from the selected model."
        self.createProblemButton.enabled = True
        parametersFormLayout.addRow(self.createProblemButton)

        #
        # displacement value
        #
        self.displacementSliderWidget = ctk.ctkSliderWidget()
        self.displacementSliderWidget.singleStep = 0.1
        self.displacementSliderWidget.minimum = -1.0
        self.displacementSliderWidget.maximum = 4.0
        self.displacementSliderWidget.value = 0.0
        self.displacementSliderWidget.setToolTip("Set displacement.")
        parametersFormLayout.addRow(
            "Displacement", self.displacementSliderWidget)

        #
        # rotation angle value
        #
        self.rotationAngleSliderWidget = ctk.ctkSliderWidget()
        self.rotationAngleSliderWidget.singleStep = 1
        self.rotationAngleSliderWidget.minimum = -90
        self.rotationAngleSliderWidget.maximum = 90
        self.rotationAngleSliderWidget.value = 0.0
        self.rotationAngleSliderWidget.setToolTip("Set rotation angle.")
        parametersFormLayout.addRow(
            "Rotation (degrees)", self.rotationAngleSliderWidget)

        #
        # body force value
        #
        self.bodyForceSliderWidget = ctk.ctkSliderWidget()
        self.bodyForceSliderWidget.singleStep = 0.1
        self.bodyForceSliderWidget.minimum = -10
        self.bodyForceSliderWidget.maximum = 10
        self.bodyForceSliderWidget.value = 0.0
        self.bodyForceSliderWidget.setToolTip("Set body force value.")
        parametersFormLayout.addRow(
            "Body Force", self.bodyForceSliderWidget)

        #
        # traction force value
        #
        self.tractionForceSliderWidget = ctk.ctkSliderWidget()
        self.tractionForceSliderWidget.singleStep = 0.1
        self.tractionForceSliderWidget.minimum = -2
        self.tractionForceSliderWidget.maximum = 2
        self.tractionForceSliderWidget.value = 0.0
        self.tractionForceSliderWidget.setToolTip("Set traction force value.")
        parametersFormLayout.addRow(
            "Traction Force", self.tractionForceSliderWidget)

        #
        # Defaults Button
        #
        self.defaultsButton = qt.QPushButton("Default parameters")
        self.defaultsButton.toolTip = "Set default parameters."
        self.defaultsButton.enabled = True
        parametersFormLayout.addRow(self.defaultsButton)

        #
        # Apply Button
        #
        self.applyButton = qt.QPushButton("Apply")
        self.applyButton.toolTip = "Run the algorithm."
        self.applyButton.enabled = False
        parametersFormLayout.addRow(self.applyButton)

        #
        # Squash and Stretch Buttons
        #
        self.squashButton = qt.QPushButton("Squash")
        self.squashButton.toolTip = "Squash"
        self.squashButton.enabled = False
        self.stretchButton = qt.QPushButton("Stretch")
        self.stretchButton.toolTip = "Stretch"
        self.stretchButton.enabled = False
        rowLayout = qt.QHBoxLayout()
        rowLayout.addWidget(self.squashButton)
        rowLayout.addWidget(self.stretchButton)
        parametersFormLayout.addRow(rowLayout)

        #
        # Rotate Buttons
        #
        self.rotateLeftButton = qt.QPushButton("Rotate Left")
        self.rotateLeftButton.toolTip = "Rotate left."
        self.rotateLeftButton.enabled = False
        self.rotateRightButton = qt.QPushButton("Rotate Right")
        self.rotateRightButton.toolTip = "Rotate right."
        self.rotateRightButton.enabled = False
        rowLayout = qt.QHBoxLayout()
        rowLayout.addWidget(self.rotateLeftButton)
        rowLayout.addWidget(self.rotateRightButton)
        parametersFormLayout.addRow(rowLayout)

        #
        # Reset Button
        #
        self.resetButton = qt.QPushButton("Reset")
        self.resetButton.toolTip = "Reset the model."
        self.resetButton.enabled = False
        parametersFormLayout.addRow(self.resetButton)

        # connections
        self.createProblemButton.connect(
            'clicked(bool)', self.onCreateProblemButton)
        self.applyButton.connect(
            'clicked(bool)', self.onApplyButton)
        self.defaultsButton.connect(
            'clicked(bool)', self.onDefaultsButton)
        self.inputSelector.connect(
            'currentNodeChanged(vtkMRMLNode*)', self.onSelect)
        self.squashButton.connect(
            'clicked(bool)', self.onSquashButton)
        self.stretchButton.connect(
            'clicked(bool)', self.onStretchButton)
        self.rotateLeftButton.connect(
            'clicked(bool)', self.onRotateLeftButton)
        self.rotateRightButton.connect(
            'clicked(bool)', self.onRotateRightButton)
        self.resetButton.connect(
            'clicked(bool)', self.onResetButton)

        # Add vertical spacer
        self.layout.addStretch(1)

        self.onSelect()

    def cleanup(self):
        pass

    def onSelect(self):
        # Refresh button states
        self.createProblemButton.enabled = self.inputSelector.currentNode()
        self.applyButton.enabled = False
        self.squashButton.enabled = False
        self.stretchButton.enabled = False
        self.rotateLeftButton.enabled = False
        self.rotateRightButton.enabled = False
        self.resetButton.enabled = False

    def onCreateProblemButton(self):
        enableScreenshotsFlag = self.enableScreenshotsFlagCheckBox.checked
        self.node = self.inputSelector.currentNode()
        mesh = create_dolfin_mesh(self.node.GetUnstructuredGrid())
        self.problem = Problem(mesh)

        self.applyButton.enabled = True
        self.squashButton.enabled = True
        self.stretchButton.enabled = True
        self.rotateLeftButton.enabled = True
        self.rotateRightButton.enabled = True
        self.resetButton.enabled = True

    def onDefaultsButton(self):
        self.displacementSliderWidget.value = 0
        self.rotationAngleSliderWidget.value = 0
        self.bodyForceSliderWidget.value = 0
        self.tractionForceSliderWidget.value = 0

    def onApplyButton(self):
        logic = FEniCS_demo_moduleLogic()
        enableScreenshotsFlag = self.enableScreenshotsFlagCheckBox.checked
        displacement = self.displacementSliderWidget.value
        rotationAngle = self.rotationAngleSliderWidget.value
        bodyForce = self.bodyForceSliderWidget.value
        tractionForce = self.tractionForceSliderWidget.value
        logic.run(self.node, self.problem,
                  displacement, rotationAngle, bodyForce, tractionForce,
                  enableScreenshotsFlag)

    def onSquashButton(self):
        self.displacementSliderWidget.value -= 0.1
        self.onApplyButton()

    def onStretchButton(self):
        self.displacementSliderWidget.value += 0.1
        self.onApplyButton()

    def onRotateLeftButton(self):
        self.rotationAngleSliderWidget.value += 5
        self.onApplyButton()

    def onRotateRightButton(self):
        self.rotationAngleSliderWidget.value -= 5
        self.onApplyButton()

    def onResetButton(self):
        self.onDefaultsButton()
        self.problem.reset()
        self.onApplyButton()


#
# FEniCS_demo_moduleLogic
#
class FEniCS_demo_moduleLogic(ScriptedLoadableModuleLogic):

    def run(self, node, problem,
            displacement, rotationAngle, bodyForce, tractionForce,
            enableScreenshots=0):
        """
        Run the actual algorithm
        """

        logging.info('Processing started')

        # Capture screenshot
        if enableScreenshots:
            self.takeScreenshot(
                'FEniCS_demo_moduleTest-Start', 'MyScreenshot', -1)

        self.update(node, problem,
                    displacement, rotationAngle,
                    bodyForce, tractionForce)

        logging.info('Processing completed')

        return True

    def update(self, node, problem,
               displacement, theta, bodyForce, tractionForce):

        # Step and solve model
        x = problem.step(displacement, 2*dolfin.pi*theta/360,
                         bodyForce, tractionForce)

        # Move the grid points
        grid = node.GetUnstructuredGrid()
        wasModifying = node.StartModify()
        dataPoints = grid.GetPoints()
        for i in range(0, dataPoints.GetNumberOfPoints()):
            dataPoints.SetPoint(i, x[i, 0], x[i, 1], x[i, 2])
            grid.SetPoints(dataPoints)
        grid.Modified()
        node.SetAndObserveMesh(grid)
        node.EndModify(wasModifying)


class FEniCS_demo_moduleTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough.
        """
        slicer.mrmlScene.Clear(0)

    def runTest(self):
        """Run as few or as many tests as needed here.
        """
        self.setUp()
        self.test_FEniCS_demo_module1()

    def test_FEniCS_demo_module1(self):
        self.delayDisplay("Starting the test")


class Problem:
    def __init__(self, mesh):

        self.mesh = mesh

        # Write mesh to file (for debugging only)
        # write_mesh(self.mesh, '/tmp/meshfromslicer.vtu')

        # define function space
        element_degree = 1
        quadrature_degree = element_degree + 1
        print("Degree of element: ", element_degree)
        print("Degree of quadrature: ", quadrature_degree)
        self.V = dolfin.VectorFunctionSpace(
            self.mesh, "Lagrange", element_degree)

        # Mark boundary subdomains
        zmin = min(self.mesh.coordinates()[:, 2])
        zmax = max(self.mesh.coordinates()[:, 2])
        print("zmin:", zmin)
        print("zmax:", zmax)
        bot = dolfin.CompiledSubDomain(
            "near(x[2], side) && on_boundary", side=zmin)
        top = dolfin.CompiledSubDomain(
            "near(x[2], side) && on_boundary", side=zmax)

        # Define Dirichlet boundary (z = 0 or z = 1)
        c = dolfin.Constant((0.0, 0.0, 0.0))
        self.r = dolfin.Expression(
            ("scale*(x0 + (x[0] - x0)*cos(theta) - (x[1] - y0)*sin(theta) - x[0])",
             "scale*(y0 + (x[0] - x0)*sin(theta) + (x[1] - y0)*cos(theta) - x[1])",
             "displacement"),
            scale=1.0, x0=0.5, y0=0.5, theta=0.0, displacement=0.0, degree=2)

        self.bcs = [
            dolfin.DirichletBC(self.V, c, bot),
            dolfin.DirichletBC(self.V, self.r, top)]

        # Define functions
        du = dolfin.TrialFunction(self.V)  # Incremental displacement
        v = dolfin.TestFunction(self.V)    # Test function
        # Displacement from previous iteration
        self.u = dolfin.Function(self.V)
        # Body force per unit volume
        self.B = dolfin.Constant((0.0, 0.0, 0.0))
        # Traction force on the boundary
        self.T = dolfin.Constant((0.0, 0.0, 0.0))

        # Kinematics
        d = len(self.u)
        I = dolfin.Identity(d)       # Identity tensor
        F = I + dolfin.grad(self.u)  # Deformation gradient
        C = F.T*F                    # Right Cauchy-Green tensor

        # Invariants of deformation tensors
        Ic = dolfin.tr(C)
        J = dolfin.det(F)

        # Elasticity parameters
        E = 10.0
        nu = 0.3
        mu = dolfin.Constant(E/(2*(1 + nu)))
        lmbda = dolfin.Constant(E * nu/((1 + nu)*(1 - 2*nu)))

        # Stored strain energy density (compressible neo-Hookean model)
        psi = (mu/2)*(Ic - 3) - mu*dolfin.ln(J) + (lmbda/2)*(dolfin.ln(J))**2

        dx = dolfin.Measure("dx", domain=mesh, metadata={
                            'quadrature_degree': quadrature_degree})
        ds = dolfin.Measure("ds", domain=mesh, metadata={
                            'quadrature_degree': quadrature_degree})
        print(dx)
        print(ds)
        Pi = psi*dx - dolfin.dot(self.B, self.u)*dx - \
            dolfin.dot(self.T, self.u)*ds
        self.F = dolfin.derivative(Pi, self.u, v)
        self.J = dolfin.derivative(self.F, self.u, du)

    def reset(self):
        """Reset solution vector."""
        self.u.vector().vec().array.fill(0.0)

    def step(self, displacement, theta, B, T):
        # Update BCs and loads
        self.r.displacement = displacement
        self.r.theta = theta
        self.B.assign(dolfin.Constant((0.0, 0.0, B)))
        self.T.assign(dolfin.Constant((0.0, 0.0, T)))

        dolfin.solve(self.F == 0, self.u, self.bcs, J=self.J,
                     form_compiler_parameters={"cpp_optimize": True,
                                               "representation": "uflacs"})

        return self.mesh.coordinates() + \
            self.u.vector()[dolfin.vertex_to_dof_map(self.V)].reshape(-1, 3)


def create_dolfin_mesh(grid):
    """
    Create a DOLFIN mesh from VTK unstructured grid.

    Args:
        grid: VTK unstructured grid.

    Returns:
        mesh (dolfin.Mesh): DOLFIN mesh object.
    """

    print("Creating DOLFIN mesh")

    v2n = vtk.util.numpy_support.vtk_to_numpy

    coords = v2n(grid.GetPoints().GetData())
    cells = v2n(grid.GetCells().GetData())
    cell_types = v2n(grid.GetCellTypesArray())
    cell_locations = v2n(grid.GetCellLocationsArray())

    num_vertices = grid.GetNumberOfPoints()
    num_cells = grid.GetNumberOfCells()

    print("VTK grid has {} vertices".format(num_vertices))
    print("VTK grid has {} cells".format(num_cells))

    mesh = dolfin.Mesh()

    vtk_to_dolfin_celltype_map = {
        # VTK_TETRAHEDRON
        10: {"celltype": "tetrahedron",
             "tdim": 3,
             "gdim": 3,
             "degree": 1,
             "ordering": [0, 1, 2, 3]},
        # VTK_HEXAHEDRON
        12: {"celltype": "hexahedron",
             "tdim": 3,
             "gdim": 3,
             "degree": 1,
             "ordering": [0, 1, 3, 2, 4, 5, 7, 6]},
    }

    # DOLFIN only supports meshes of a single element type
    if not (cell_types == cell_types[0]).all():
        raise ValueError("Grid contains cells of different types")

    dolfin_cell_data = vtk_to_dolfin_celltype_map[cell_types[0]]

    celltype = dolfin_cell_data["celltype"]
    tdim = dolfin_cell_data["tdim"]
    gdim = dolfin_cell_data["gdim"]
    degree = dolfin_cell_data["degree"]

    me = dolfin.MeshEditor()
    me.open(mesh, celltype, tdim, gdim, degree=degree)

    me.init_vertices(num_vertices)
    for i, x in enumerate(coords):
        me.add_vertex(i, x)

    me.init_cells(num_cells)
    for i, l in enumerate(cell_locations):
        cell = cells[l+1:l+1+cells[l]]
        cell = [cell[j] for j in dolfin_cell_data["ordering"]]
        me.add_cell(i, cell)

    # mesh.order()
    mesh.init()

    print("DOLFIN mesh has {} vertices".format(mesh.num_vertices()))
    print("DOLFIN mesh has {} cells".format(mesh.num_cells()))

    return mesh


# For debugging

def create_box_mesh(cell_type):
    p0 = dolfin.Point(0, 0, 0)
    p1 = dolfin.Point(1, 1, 1)
    cell_types = {
        "hexahedron": dolfin.CellType.Type.hexahedron,
        "tetrahedron": dolfin.CellType.Type.tetrahedron}
    mesh = dolfin.BoxMesh.create([p0, p1], [8, 8, 8],
                                 cell_type=cell_types[cell_type])
    return mesh


def write_mesh(mesh, filename):
    vtkfile = dolfin.VTKFile(filename, "ascii")
    vtkfile.write(mesh)


def create_and_write_meshes():
    mesh = create_box_mesh("hexahedron")
    write_mesh(mesh, "/tmp/cube-hex.pvd")
    mesh = create_box_mesh("tetrahedron")
    write_mesh(mesh, "/tmp/cube-tet.pvd")
