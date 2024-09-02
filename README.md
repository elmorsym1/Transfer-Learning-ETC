# Enhancing Effective Thermal Conductivity Predictions in Digital Porous Media Using Transfer Learning

**Data:**

The data includes sub-volumes of the following rocks,
  - Bentheimer Sandstone 
  - Ketton Limestone
  - Berea Sandstone

The raw CT rock cores are obtained from the [Imperial Colloge London portal](https://www.imperial.ac.uk/earth-science/research/research-groups/pore-scale-modelling/micro-ct-images-and-networks/).

The sub-volumes are simulated for Effective Thermal Conductivity using OpenFOAM and their results are summerized in the provided excel sheet having the following information,

 - Number of sub-samples = 40,041
 - Labels description:
    - casename = sub-sampling index per rock type sample
    - porosity = ratio of void fraction
    - esv = effective of soil volume fraction
    - rock_type = 
                   
                   {
      
                   1:Berea Sandstone,
                   
                   2:Bentheimer Sandstone,
                   
                   3:Ketton Limestone,
      
                   }
                   
    - k = Effective Thermal Conductivity
  
**Software and Libraries:**

Numerical simulations of the 3D porous media samples were conducted using OpenFOAM®, which is an open-source set of solvers for CFD simulations (Horgue et al., 2015). The analytical solutions are developed using the symbolic computation solver SymPy 1.10.1 in Python 3.9.12. We train the machine learning model using the open-source software interface Keras 2.4.0 and TensorFlow 2.3.1 on NVIDIA GeForce RTX 2080 Ti GPUs. Figures were made with Matplotlib 3.5.1, available under the Matplotlib license at https://matplotlib.org/. 


This work has been submitted to Computers and Geosciences journal, and still under review for publication. Please contact repository owner for a written permission before use.

Paper link: To be provided post official publication.

For more information, please contact the repository owner at: elmorsym777@gmail.com


