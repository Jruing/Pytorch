{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np \n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "x = np.linspace(-5, 5, 1000)\n",
    "\n",
    "y = lambda x:((x+1)**2+4) * (x<0) + (x>=0) * (4*(x-1)**2+1)\n",
    "plt.scatter([-1,1], [4,1], c='r') # 极值点\n",
    "plt.plot(x, y(x));"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "x0 = -4\n",
    "lr = 0.01\n",
    "\n",
    "def partial(x):\n",
    "    if x == 0:\n",
    "        raise ValueError('不可导')\n",
    "    dx = (2*(x+1)) * (x<0) + (8*(x-1)) * (x>0)\n",
    "    return dx\n",
    "\n",
    "for i in range(10000):\n",
    "    x0 -= lr * partial(x0)\n",
    "    plt.scatter(x0, y(x0), c='y')\n",
    "    \n",
    "plt.scatter([-1,1], [4,1], c='r')    \n",
    "plt.plot(x, y(x));"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import copy\n",
    "\n",
    "x0 = -4\n",
    "lr = 0.2\n",
    "alpha = 0.1\n",
    "dx_old = 0\n",
    "for i in range(100):\n",
    "    dx = alpha * partial(x0) + dx_old * (1-alpha)\n",
    "    x0 -= lr * dx\n",
    "    plt.scatter(x0, y(x0), c='y')\n",
    "    \n",
    "    dx_old = copy.copy(dx)\n",
    "    \n",
    "plt.scatter([-1,1], [4,1], c='r')  \n",
    "plt.plot(x, y(x));"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
	",
    "$$\\gamma_k = \\sum_{i=1}^{k}\\nabla \\theta_i^2 \\\\ \\theta_k = \\theta_{k-1} - \\frac{lr}{\\sqrt{\\gamma_k}+\\epsilon}\\nabla \\theta_k$$"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### RMSProp\n",
    "RMSProp则是对Adagrad进行了EMA, \n",
    "$$\\gamma_k = \\alpha \\gamma_{k-1} + (1 - \\alpha) \\nabla \\theta_k \\\\ \\theta_k = \\theta_{k-1} - \\frac{lr}{\\sqrt{\\gamma_k}+\\epsilon}\\nabla \\theta_k$$"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "$$m_t = \\beta_1 m_{t-1} + (1 - \\beta_1)\\nabla \\theta_t \\\\ v_t = \\beta_2 v_{t-1} + (1-\\beta_2)\\nabla\\theta^2_t$$\n",
    "$\\frac{1}{1-\\beta_i^t}$\n",
    "$$\\hat{m_t} = \\frac{m_t}{1-\\beta_1} \\\\ \\hat{v_t} = \\frac{v_t}{1-\\beta_2}$$\n",
    "参数更新\n",
    "$$\\theta_t = \\theta_{t-1} - lr  \\frac{\\hat{m_t}}{\\sqrt{\\hat{v_t}}+\\epsilon}$$"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### PyTorch优化"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.datasets import make_classification\n",
    "data = make_classification(n_samples=1000, n_features=20,)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch as t\n",
    "\n",
    "x = t.from_numpy(data[0]).float()\n",
    "x.requires_grad = True\n",
    "y_ = t.from_numpy(data[1]).float()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "class MLP(t.nn.Module):\n",
    "    def __init__(self, \n",
    "                 num_classes=10):\n",
    "        super(MLP, self).__init__()\n",
    "        self.linear1 = t.nn.Linear(in_features=20, out_features=32)\n",
    "        self.bn1 = t.nn.BatchNorm1d(32)\n",
    "        self.linear2 = t.nn.Linear(32,64)\n",
    "        self.bn2 = t.nn.BatchNorm1d(64)\n",
    "        self.linear3 = t.nn.Linear(64,1)\n",
    "    \n",
    "    def forward(self, x):\n",
    "        x = self.linear1(x)\n",
    "        x = self.bn1(x)\n",
    "        x = t.relu(x)\n",
    "        x = self.linear2(x)\n",
    "        x = self.bn2(x)\n",
    "        x = self.linear3(x)\n",
    "        x = t.sigmoid(x)\n",
    "        return x\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "criterion = t.nn.BCELoss()\n",
    "\n",
    "models = []\n",
    "for i in range(5):\n",
    "    models.append(MLP())\n",
    "opt1 = t.optim.SGD(models[0].parameters(), lr=1e-3)\n",
    "opt2 = t.optim.Adagrad(models[1].parameters(), lr=1e-3)\n",
    "opt3 = t.optim.RMSprop(models[2].parameters(), lr=1e-3)\n",
    "opt4 = t.optim.Adam(models[3].parameters(), lr=1e-3)\n",
    "opt5 = t.optim.SGD(models[4].parameters(), lr=1e-3, momentum=0.9)\n",
    "opts = [opt1, opt2, opt3, opt4, opt5]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "losses = []\n",
    "for i, opt in enumerate(opts):\n",
    "    \n",
    "    losses.append([])\n",
    "    for epoch in range(100):\n",
    "        opt.zero_grad()\n",
    "        \n",
    "        y = models[i](x)\n",
    "        \n",
    "        loss = criterion(y.squeeze(), y_)\n",
    "        loss.backward()\n",
    "        \n",
    "        opt.step()\n",
    "        losses[i].append(loss.data.item())\n",
    "        "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "cmap = ['k', 'r', 'b','y','cyan']\n",
    "for loss, c in zip(losses, cmap):\n",
    "    plt.plot(range(100), loss, c=c)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "tf",
   "language": "python",
   "name": "tf"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}