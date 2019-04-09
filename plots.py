# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# %%
x = np.linspace(-2, 3, num=15)
dense_x = np.linspace(-2, 3, num=100)
y = x**2 + 0.7 * np.random.randn(*x.shape)
#%%
p1 = np.polyfit(x, y, 1)
p2 = np.polyfit(x, y, 2)
p12 = np.polyfit(x, y, 12)
#%%
plt.rc('font', size=11)

plt.figure(1, figsize=(11,3), dpi=300)
plt.subplot(131)
plt.title(" deg=1 (underfitting)")
plt.plot(x, y, 'o')
plt.plot(dense_x, np.polyval(p1, dense_x))

plt.subplot(132)
plt.title(" deg=2 (optimal)")
plt.plot(x, y, 'o')
plt.plot(dense_x, np.polyval(p2, dense_x))

plt.subplot(133)
plt.title("deg=12 (overfitting)")
plt.plot(x, y, 'o')
plt.plot(dense_x, np.polyval(p12, dense_x))

plt.savefig("polyfit.png", bbox_inches='tight')
plt.show()
#%%

A = np.array([[1, 0], [0, 0], [0, 0]])
B = np.array([[1, 0], [0, 0], [0, 0.0001]])
#%%
np.linalg.pinv(A)
#%%
np.linalg.pinv(B)

# %%
plt.figure(figsize=(5, 2), dpi=300)
x = np.linspace(-1, 1, num=15) + 0.1* np.random.rand(*x.shape)
y = np.where(x > 0, np.ones_like(x), np.zeros_like(x)) + 0.05 * np.random.randn(*x.shape)
model = np.polyval(np.polyfit(x, y, 1), x)
plt.plot(x, y, 'o')
# plt.plot([-1, -0.05, 0.05, 1], [0, 0, 1, 1], '--', label='step function')
plt.plot(x, model, '-')
plt.savefig('step_linfit.png', bbox_inches='tight')
#%%


#%%
