<p align="center"
    <img width="460" height="460" src="/Plots/sl3(C).gif">
</p>
   
# Eigenvalues of the Adjoint Representation

This program attempts to provide a unique visualization of the root space of a semi-simple Lie algebra. Namely, the Special linear Lie algebra. This Lie algebra is the set of all nxn matrices over the complex numbers, with trace 0. The images produced are composed of pixels whose coordinates are derived from Eigenvalues, and whose color is derived from the angle the cooresponding Eigenvector has with a fixed vector. For a more formal explanation of the math behind this program see [DETAILS](/DETAILS.pdf).

It is worth taking a quick detour to see the number of computations neccessary depending on what DIM and COEFFS are set to. This should inform your choice of the two parameters with respect to the hardware you will be running the program on, and the performance expectations therein. 

## Performance

Letting DIM = 10 and COEFFS = 100. We begin by calculating the dimension of the Cartan subalgebra. First we must see how the basis of our Lie algebra is calculated. Here we have that `self.mat_dim = DIM`.

```python
basis_index = 0
for i in range(self.mat_dim):
    for j in range(self.mat_dim):
        if (i != j):
            self.basis[basis_index][i][j] = complex(1, 0)
            basis_index += 1

for i in range(self.mat_dim - 1):
    self.basis[basis_index][i][i] = complex(1, 0)
    self.basis[basis_index][i+1][i+1] = complex(-1, 0)
    basis_index += 1
```

What the above code is accomplishing is the following: we are creating the set of all     10 x 10 matrices whose entries fall into two categories

1. For every entry not on the principle diagonal, we create a matrix such that it has a 1 in that entry. For a 10 x 10 matrix, there are  (10)(10) - 10 = 90 such entries not on the principle diagonal, and thus we create 90 such matrices.

2. Starting from the (0, 0) entry, we place a 1 and on the (1, 1), we place a -1. Doing this for every entry along the priciple diagonal, we obtain 10 - 1 = 9 such matrices. 

Taking these two categories together, we have a basis for our Lie algebra. Moreover, considering the number of basis elements we have, it is clear that the Lie algebra is 99 dimensional.

        A Cartan subalgebra (with respect to the Special linear Lie algebra) can be obtained simply by taking the span of the basis elements that are of category 2. Therefore, the dimension of this Cartan subalgebra is 9. 

        The next thing the program does is to calculate the adjoint representation of these basis elements. This calculation is covered in detail in [DETAILS](/DETAILS.pdf). The result of this calculation is a set of 9  99x99 matrices. What we will do with this set is create every possible linear combination of them using our COEFFS = 100 coefficients. 

        This final calculation is to be the takeaway of this section. Given 9 basis elements, and 100 coeffcients, there are 100^9 = 1,000,000,000,000,000,000 many linear combinations. Therefore, the choice of DIM and COEFFS need to be made with care and consideration to your system. 

---
  

                         

