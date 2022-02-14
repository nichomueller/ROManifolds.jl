---
   title: Forward and Reverse mode in automatic differentiation
   author: Santiago Badia
---

# Forward and reverse model in AD

## Short intro to differential geometry

Let $X$  and $Y$ be two manifolds and $\Phi : X \rightarrow Y$ a map among them.

At each point $x \in X$, we have the tangent space $T_x X$. A 1-form $\omega_X$ is such that $\omega_X(x) : T_x X \rightarrow \mathbb{R}$.

For simplicity, let us assume that $X$ is flat. We informally define the derivative of the map as
$$
D_x \Phi(v) = \Phi(x+v) - \Phi(x)
$$
in the infinitesimal limit, for any $v \in T_x X$.

The **pushforward** operator $\Phi^*(x): T_x X \rightarrow T_{\Phi(x)}Y$ at $x \in X$  is simply
$$
\Phi^*(x)(v) \doteq D_X \Phi(x)(v), \qquad x \in X, \ v \in T_x X.
$$
In other words, the pushforward is the derivative of the map $\Phi$. If we fix a basis $\left\{ x_1, \ldots, x_d \right\}$ for $X$, the pushforward operator can be represented as a matrix, the _Jacobian_ matrix. Its application over a tangent vector is a Jacobian-vector multiplication. In this case, we can write it as:
$$
\Phi^*(x)(v) \doteq D_X \Phi(x)(v) = v^i \partial_{x_i} \Phi, \qquad x \in X, \ v \in T_x X.
$$
The tangent space basis for $TX$ is $\left\{\partial_{x_1}, \ldots, \partial_{x_n}\right\}$. 

The tangent space basis for $TX$ is $\left\{\partial_{x_1}, \ldots, \partial_{x_n}\right\}$. The tangent space for $T_yY$ is $\left\{\partial_{y_1}, \ldots, \partial_{y_n}\right\}$. 

Let us consider a 1-form $\omega_Y: T_y Y \rightarrow \mathbb{R}$. We can now define a 1-form in $X$, the **pullback** $\Phi_*(w_Y):T_x X \rightarrow \mathbb{R}$ as 
$$
\Phi_*(\omega_Y)|_x(v) \doteq \omega_Y|_{\Phi(x)} (\Phi^*(v)), \qquad x \in X, \ v \in Tx X.
$$
In terms of a basis, it can be expressed as the Jacobian-transpose-vector multiplication (remind that a 1-form at each point provides a covector, i.e., a row vector).

## Automatic differentiation: two modes

Let us consider the function composition 
$$
\varphi \doteq \varphi_L \circ \varphi_{L-1} \circ \ldots \varphi_0 : X_0 \rightarrow X_{L+1}, 
$$
where 
$$
\varphi_i: X_i \rightarrow X_{i+1}, \qquad i = 0, \ldots, L.
$$
Let us pick $w_0 \in X_0$. We aim at computing $\varphi(w_0)$ and $d_{X_0} \varphi(w_0)$. 

### Option 1: Forward differentation

We compute the derivatives as:
$$
\dot{\varphi}_i|_{w_i} = D_{X_{i}} \varphi_i(w_{i}), \qquad  w_{i+1} = \varphi_i(w_{i}), \qquad i=0,\ldots,L.
$$
We move from level 0 to the final level and compute both the function and its derivative at each level at the same time. After this process, we recover
$$
\dot{\varphi}|_{w_0} = D_{X_0} \varphi(w_0) = \dot{\varphi}_L|_{w_L}, \qquad \varphi(w_0) = w_{L+1}.
$$


### Option 2: Backward differentiation

Using the chain rule, we get:
$$
D_{X_0} \varphi(w_0) = D_{X_{L}} \varphi_{L}|_{w_{L}} D_{X_{L-1}} \varphi_{L-1}|_{w_{L-1}} \ldots D_{X_{0}} \varphi_{0}|_{w_{0}}.
$$
Let us define the adjoint as:
$$
\bar{\varphi}_{L+1} = 1_{X_{L+1}}, \qquad  \bar{\varphi}_i|_{w_{i}} = \bar{\varphi}_{i+1} D_{X_i} \varphi_i|_{w_{i}}, \qquad i = L, \ldots, 0
$$
We can easily check that:
$$
D_{X_0} \varphi(w_0) = \bar{\varphi}_0|_{w_0}.
$$
On the hand, we can see that in fact $\bar{\varphi_i}$

Let us consider a 1-form $\omega_Y: T_y Y \rightarrow \mathbb{R}$. We can now define a 1-form in $X$, the **pullback** $\Phi_*(w_Y):T_x X \rightarrow \mathbb{R}$ as 
$$
\Phi_*(\omega_Y)|_x(v) \doteq \omega_Y|_{\Phi(x)} (\Phi^*(v)), \qquad x \in X, \ v \in Tx X.
$$

## Reverse differentiation and the pullback

We note that the derivative of a function in $X$ is a 1-form. Let us consider the case for $L=1$ for simplicity. The general case can be obtained by composition.   

Using the chain rule, we get:
$$
D_{X_0} \varphi |_{w_0} = D_{X_1} \varphi_1 |_{\Phi(w_0)}  D_{X_{0}} \varphi_{0}|_{w_{0}}.
$$
Using the definition of the pullback for the map $\varphi_0$  we get:
$$
\varphi_0^*(D_{X_1} \varphi_1)|_{w_0} = D_{X_1} \varphi_1 |_{\Phi(w_0)} D_{X_0} \varphi_0 |_{w_0}.
$$
Thus,
$$
D_{X_0} \varphi |_{w_0} =  \varphi_0^*(D_{X_1} \varphi_1)|_{w_0}.
$$
We note that for 0-forms the pullback is simply:
$$
\varphi_0^*(\varphi_1) =  \varphi_1 \circ \varphi_0 = \varphi.
$$
Thus, we have proved that
$$
D_{X_0} \varphi_0^*(\varphi_1) |_{w_0} =  \varphi_0^*(D_{X_1} \varphi_1)|_{w_0},
$$
i.e., the derivative and pullback commute.

We can observe that the adjoint at a given stage is the pullback of the derivative at the previous stage. 

## Matrix interpretation

At the end of the day, we must fix bases for our spaces, since we have to implement things. In this case, we can write derivatives using Jacobian matrices. Let us define

$$
J^{i} = D_{X_i} \varphi_i \in \mathbb{R}^{N_{i+1}\times N_{i}},
$$
where $N_i$ is the dimension of the ambient space of $X_i$. We want to compute
$$
D_{X_0} \varphi = D_{X_L} \varphi^{L} D_{X_{L-1}} \varphi^{L-1} \ldots D_{X_0} \varphi^0 = J^{L} J^{L-1} \ldots J^0 \in \mathbb{R}^{N_L\times N_0}
$$
Using the forward propagation, we compute $J^0$ and propagate to higher levels by Jacobian multiplication, i.e., pushing forward the gradient vector.

Instead, using backward propagation, we start from the last level in the chain, computing $J^L$, and multiply in the reverse direction, i.e., pulling back derivatives.

Usually, $N_L = 1$ (the target function is a scalar function). $J^L$ is a row vector and the product is just a Jacobian-transpose-vector product. This is the case during the whole process. That is the reason why backpropagation is more efficient when the output dimension is low compared to the input dimension $N_0$. In the opposite direction, we would have to multiply at each level against $N_0$ vectors. Certainly, if $N_0 >> N_L$, this is costly. 















