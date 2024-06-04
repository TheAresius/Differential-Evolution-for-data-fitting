# Differential Evolution for data fitting
## Mathematical description
This script uses the differential evolution model to search for parameters that yield the global minimum of a given error function.

I was inspired by [this](https://sci-hub.st/https://royalsocietypublishing.org/doi/abs/10.1098/rsta.1999.0469) article and some changes were made to the error function in order to incorporate experimental data uncertainty into the model.

The error function that the algorithm aims to minimize is:
<p align=center>
  $E\left(\mathbf{M},\mathbf{P}\right) = \frac{1}{N-1} \sum r\left(\mathbf{M,P}\right)$
</p>

where $\mathbf{M}$ is a $4\times N$ matrix with the $N$ experimental data points:

<p align=center>
  $$
  \begin{bmatrix}
  x_1 &x_2 &\cdots &x_i &\cdots &x_N \\
  y_1 &y_2 &\cdots &y_i &\cdots &y_N \\
  \sigma_{x_1} &\sigma_{x_2}&\cdots &\sigma_{x_i} &\cdots &\sigma_{x_N} \\
  \sigma_{y_1} &\sigma_{y_2}&\cdots&\sigma_{y_i}&\cdots &\sigma_{y_N}
  \end{bmatrix}
  $$
</p>

$\mathbf{P}$ is the parameter vector containing the best-so-far parameter values for the physical model $f\left(\mathbf{M,P}\right)$ describing the observed phenomena and $r\left(\mathbf{M,P}\right)$ is the residual function that is adapted depending on the nature of the problem.
For the particular problem our group analysed we use the residual function as:
<p align=center>
$r\left(\mathbf{M,P}\right)=r_{|log|}=\left| \ln \left( \left| y_i \right| \right) - \ln \left( \left| f(x_i; \mathbf{P}) \right| \right) \right|$
</p>

I added a term to the error function as follows:

<p align=center>
  $E\left(\mathbf{M},\mathbf{P}\right) = \frac{1}{N-1} \sum \frac{r\left(\mathbf{M,P}\right)}{c}$
</p>

where c is the experimental uncertainty contribution:
<p align=center>
  $\sqrt{\sigma_{y_i}^2 + \left(\sigma_{x_i} \frac{d}{dx} f\left(x_i; \mathbf{P}\right)\right)^2}$
</p>

The term $c$ takes into account the changes $\sigma_{y_i}$ and $\sigma_{x_i}$ into the variables $y$ and $x$. A small change due to $\sigma_{y_i}$ yields a change $\frac{dy}{dy}\sigma_{y_i}=\sigma_{y_i}$ into the $y$ variable, while a small change due to $\sigma_{x_i}$ yields a change $\frac{dy}{dx}\sigma_{x_i}$ into the $y$ variable, which depends on $x$.
Instead of using an analytical derivative I used a numerical one, since it could very computationally costly to find an analytical derivative for a more complex physical model. The numerical derivative used is a simple finite differences:
```
def numerical_derivative(x, params, h=1e-5):
    y = (model(x + h, params) - model(x - h, params)) / (2 * h)
    return y
```
For further reading, check out the article linked above. See also "Differential Evolution: A Practical Approach to Global Optimization" by Kenneth V. Price, Rainer M. Storn, and Jouni A. Lampinen (Springer, 2005).
## A real life problem
Our group used optical spectroscopy to study the emission spectrum of sodium, including the sharp, principal, and diffuse series, as well as the Rydberg constant and the quantum defects associated with sodium.

The equations that describe the wavelength $\lambda$ of sodium with respect to its principal quantum number $n$ are:

sharp series:
<p align=center>
    $\frac{1}{\lambda}=\frac{R}{(3-\mu_p)^2}-\frac{R}{(n-\mu_s)^2}$
</p>
principal series:
<p align=center>
    $\frac{1}{\lambda}=\frac{R}{(3-\mu_s)^2}-\frac{R}{(n-\mu_p)^2}$
</p>
diffuse series:
<p align=center>
    $\frac{1}{\lambda}=\frac{R}{(3-\mu_p)^2}-\frac{R}{(n-\mu_d)^2}$
</p>

where $R$ is the Rydberg's constants and $\mu_s$, $\mu_p$, $\mu_d$ are the quantum deffects for sodium.

Since it is quite difficult to obtain a large amount of data for each series (we tipically obtain the number of data points equal to the number of parameters), the usual fitting procedure yields very poor results. To maximize the utility of our experimental data, we adopted an alternative method.

By combining the equations into a coupled equation with shared parameters we can fit all parameters simultaneously. The criteria for combining the equations is:

<p align=center>
$$
    \frac{1}{\lambda}=
    \left\{ \begin{array}{cll}
    \frac{R}{\left(3-\mu_p\right)^2}-\frac{R}{\left(x-\mu_s\right)^2} & : \ x < 10 & (\text{sharp series})\\
    \frac{R}{\left(3-\mu_s\right)^2}-\frac{R}{\left((x-10)-\mu_p\right)^2} & : \ 10\leq x < 20 & (\text{principal series})\\
    \frac{R}{\left(3-\mu_p\right)^2}-\frac{R}{\left((x-20)-\mu_d\right)^2} & : \ x \geq 20 & (\text{diffuse series})
    \end{array} \right.
$$
</p>

where $x$ is the corrected principal quantum number. It was added $0$, $10$ or $20$ to the principal quantum number depending on which series the data belongs.

This method was incorporated using nested numpy.where:
```
def model(x, param):
    s, p, d, R = param
    y = np.where(x<10, R*(1/((3-p)**2) - 1/((x-s)**2)),
                np.where(x<20, R*(1/((3-s))**2 - 1/(x-10-p)**2),
                        R*(1/((3-p)**2) - 1/((x-20-d)**2))))  
    return y
```
Theoretical values for the parameters are widely known. The expected value for the Rydberg constant was $R=1.0973731568157 (12)\times10^{-7}$ $m^{-1}$ (NIST) and the expected values for the quantum corrections were $\mu_s=1.348$, $\mu_p=0.855$ e $\mu_d=0.0148$ (doi:10.1103/physreva.18.229).
# Results
We have obtained $R=1.09365 (81)\times 10^{-7}$ $m^{-1}$, $\mu_s=1.3686 (45)$, $\mu_p=0.87335 (36)$ e $\mu_d=0.0150(67)$ which are excelent results given the limitation of our data!
![alt text](https://github.com/TheAresius/Differential-Evolution-for-data-fitting/blob/main/graph.png?raw=true)
