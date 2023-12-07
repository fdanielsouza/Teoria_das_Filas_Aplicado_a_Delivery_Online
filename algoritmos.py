#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
import itertools as it
from operator import itemgetter
from scipy.special import factorial
from scipy.stats import norm, poisson, gamma
from scipy.optimize import minimize, Bounds
from scipy.integrate import quad


# ### Funções para estimar a taxa de chegadas não homogêneas

# In[ ]:


def Y_t(x, T):
    """
    Contagem de eventos não ocorridos em \[x, T\]
    
    Args:
        x (float): momento do evento x pertencente à sequência de momentos \[0, T\]
        T (list): lista de eventos de um processo estocástico empírico
        
    Returns:
        int: quantidade de eventos que ocorrerão entre \(x, T\]
    """
    return len(T[T >= x])


def epanechnikov(x):
    """
    Função de Kernel de Epanechnikov
    
    Args:
        x (float): valor da distribuição empírica
        
    Returns:
        float: valor suavizado pelo Kernel
    """
    return 3 / 4 * (1 - x ** 2) if np.abs(x) <= 1 else np.zeros(1)

    
def alfa_t(x, T, b, K=norm.pdf):
    """
    Estimador do valor de â(t), ou a intensidade de ocorrência dos eventos em t
    
    Args:
        x (float): momento do evento x pertencente à sequência de momentos \[0, T\]
        T (list): lista de eventos de um processo estocástico empírico
        b (float): parâmetro de suavização do Kernel
        K (function, optional): função de Kernel, por padrão é o Kernel Gaussiano
        
    Returns:
        float: valor estimado da intensidade a(t)
    """
    b = np.array([b]) if not isinstance(b, np.ndarray) else b
    n = len(T)
    K_t = [K((x - Ti) / b) / Y_t(Ti, T) for i, Ti in enumerate(T)]
    return np.sum(K_t) / b


def Lambda_t(x, T, b, K=norm.pdf):
    """
    Processo de contagem da intensidade estimada â(t)
    
    Args:
        x (float): momento do evento x pertencente à sequência de momentos \[0, T\]
        T (list): lista de eventos de um processo estocástico empírico
        b (float): parâmetro de suavização do Kernel
        K (function, optional): função de Kernel, por padrão é o Kernel Gaussiano
        
    Returns:
        float: quantidade de eventos ocorridos no momento x dada a intensidade estimada â(t)
    """
    return alfa_t(x, T, b, K) * Y_t(x, T)


def A_t(x, T, b, K=norm.pdf):
    """
    Integração do processo λ para obter o processo Λ
    
    Args:
        x (float): momento do evento x pertencente à sequência de momentos \[0, T\]
        T (list): lista de eventos de um processo estocástico empírico
        b (float): parâmetro de suavização do Kernel
        K (function, optional): função de Kernel, por padrão é o Kernel Gaussiano
        
    Returns:
        float: quantidade de eventos acumulados entre \[0, x\) dada a intensidade estimada â(t)
    """
    return quad(Lambda_t, 0, x, (T, b, K))


def A_t_parcial(xi, xj, T, b, K=norm.pdf):
    """
    Integração parcial do processo λ para obter valores somáveis processo Λ de modo mais rápido
    
    Args:
        xi (float): momento do evento x(t-1) pertencente à sequência de momentos \[0, T\]
        xj (float): momento do evento x(t) pertencente à sequência de momentos \[0, T\]
        T (list): lista de eventos de um processo estocástico empírico
        b (float): parâmetro de suavização do Kernel
        K (function, optional): função de Kernel, por padrão é o Kernel Gaussiano
        
    Returns:
        float: quantidade de eventos acumulados entre \[xi, xj\) dada a intensidade estimada â(t)
    """
    return quad(Lambda_t, xj, xi, (T, b, K))


def A_t_Riemann(T, b, K=norm.pdf, h=0.0001):
    """
    Aproximação de integração do processo λ para obter valores somáveis processo Λ através de Somas de Riemann
    
    Args:
        T (list): lista de eventos de um processo estocástico empírico
        b (float): parâmetro de suavização do Kernel
        K (function, optional): função de Kernel, por padrão é o Kernel Gaussiano
        h (float, optional): valor de largura para incrementos de somas 
        
    Returns:
        list: lista de valores aproximados do processo Λ dada a intensidade estimada â(t)
    """
    oT = np.insert(np.sort(T), 0, 0)
    return oT[1:], h * np.cumsum([np.sum([Lambda_t(x, T, b, K) for x in np.arange(i, j, h)]) for i, j in zip(oT[:-1], oT[1:])])


def cv_loglike(b, x, K=norm.pdf):
    """
    Encontra valor da validação cruzada da log-verossimilhança para o parâmetro de suavização b
    
    Args:
        b (float): valor do parâmetro b para referência
        x (list): lista de eventos de um processo estocástico empírico
        K (function, optional): função de Kernel, por padrão é o Kernel Gaussiano
        
    Returns:
        float: valor da log-verossimilhança
    """
    n = len(x)
    total_loglike = 0
    
    for i in range(n):
        x_sem_i = np.delete(x, i)
        xi = x[i]
        alfa_fxi = alfa_t(xi, x_sem_i, b, K)
        if alfa_fxi <= 0: continue
        total_loglike += np.log(alfa_fxi)
        
    return -total_loglike


def b_otimo_cv(f, ini, params, lims):
    """
    Busca o valor ótimo do parâmetro b através da minimização da log-verossimilhança
    
    Args:
        f (function): função para minimizar
        ini (float): valor inicial do parâmetro b
        params (tuple): parâmetros adicionais da função f
        lims (tuple): limites do parâmetro ini para o processo de otimização
        
    Returns:
        np.ndarray: valor otimizado do retorno da função
    """
    return minimize(f, (ini,), params, bounds=lims)['x']


# ### Funções de estimativas de probabilidades

# In[ ]:


def V(x, lamb):
    """
    Calcula a probabilidade de chegar x clientes na fila, dada uma taxa de chegadas lamb
    
    Args:
        x (int): quantidade de clientes que chegam na fila
        lamb (float): taxa média de cehgadas
        
    Returns:
        float: probabilidade de chegada de clientes, usando a distribuição de Poisson
    """    
    return poisson.pmf(x, lamb)


def S(x, k, theta):
    """
    Calcula a probabilidade de um atendimento durar x unidades de tempo de serviço, dados os parâmetros k e theta
    
    Args:
        x (int): quantidade de unidades de tempo de serviço discreta
        k (float): parâmetro de forma da distribuição Gama
        theta (float): parâmetro de escala da distribuição Gama
        
    Returns:
        float: probabilidade de um atendimento durar x unidades de tempo de serviço, usando a distribuição Gama
    """    
    return gamma.cdf(x, k, loc=0, scale=theta) - gamma.cdf(x - 1, k, loc=0, scale=theta)


# ### Funções para calcular as probabilidades dos estados da filas

# In[ ]:


def remover_tempos_servico(vetores_n_x, C):
    """
    Cria o conjunto de vetores (n1: ym, ym-1, ... y1) a partir dos vetores (n: xm, xm-1, ..., x1)
    
    Args:
        vetores_nx (iterator): iterador na forma n; xm, xm-1, ... x1; p(xm, xm-1, ..., x1)
        C (int): quantidade de servidores ativos
        
    Yields:
        tuple: iterador na forma n1; ym, ym-1, ... y1; a; p(xm, xm-1, ..., x1)
    """
    for n, xm_xi_x1, p_xm_xi_x1 in vetores_n_x:
        rems = np.maximum(xm_xi_x1 - C, 0)
        movs = np.array([np.pad(xi, (1, 0), 'constant', constant_values=0)[:-1] for xi in np.minimum(xm_xi_x1, C)])
        ym_yi_y1 = rems + movs
        n1 = n - np.minimum(xm_xi_x1[:,-1], C).astype(int)
        a = np.sum(ym_yi_y1, axis=1).astype(int)
        
        for vetor_n1_yi_a_px in zip(*(n1, ym_yi_y1, a, p_xm_xi_x1)):
            yield vetor_n1_yi_a_px


def ordenar_n1(vetores_n1_y_a_px):
    """
    Agrupa e ordena os vetores (n1: ym, ym-1, ... y1) para melhorar a eficiência dos loops
    
    Args:
        vetores_n1_y_a_px (iterator): iterador na forma n1; ym, ym-1, ... y1; a; p(xm, xm-1, ..., x1)
        
    Yields:
        tuple: vetores agrupados utilizando n1 como chave
    """
    for k, g in it.groupby(sorted(vetores_n1_y_a_px, key=itemgetter(0)), lambda x: x[0]):
        yield k, [(v[1], v[2], v[3]) for v in g]

        
def gerar_vetores_z(nnewst, m):
    """
    Obtém todas as combinações de vetores nnewst = (zm, zm-1, ... z1)
    
    Args:
        nnewst (int): número de novos clientes entrando em atendimento
        m (int): valor máximo possível de unidades de tempos de serviço
        
    Returns:
        np.ndarray: conjunto de vetores (zm, zm-1, ..., z1)
    """
    def gerar_zm_zi_z1(zj_z1, zm_zi_z1, st):
        if zj_z1 == 0 and st == m:
            z_combs.append(zm_zi_z1.copy())
            return 
        if st == m:
            return
        for zj in range(zj_z1 + 1):
            zm_zi_z1[st] = zj
            gerar_zm_zi_z1(zj_z1 - zj, zm_zi_z1, st + 1)
            zm_zi_z1[st] = 0

    z_combs = []
    zm_zi_z1 = np.zeros(m, dtype=int)
    gerar_zm_zi_z1(nnewst, zm_zi_z1, 0)
    
    return np.array(z_combs)


def gerar_probs_z(nnewst, vetores_z, m, k, theta):
    """
    Obtém as probabilidades para os vetores (zm, zm-1, ... z1)
    
    Args:
        nnewst (int): número de novos clientes entrando em atendimento
        vetores_z (np.ndarray): conjunto de vetores (zm, zm-1, ... z1)
        m (int): valor máximo possível de unidades de tempos de serviço
        k (float): parâmetro de forma da distribuição Gama
        theta (flaot): parâmetro de escala da distribuição Gama
        
    Returns:
        np.ndarray: conjunto de vetores de probabilidades p(zm, zm-1, ..., z1)
    """
    probs_ts = np.array([S(m - mi, k, theta) for mi in range(m)])
    
    return nnewst / np.prod(factorial(vetores_z), axis=1) * np.prod(probs_ts ** vetores_z, axis=1)


def gerar_vetores_w(vetores_n_x, L, C, m, lamb, k, theta):
    """
    Obtém os vetores (n2: wm, wm-1, ... w1)
    
    Args:
        vetores_n_x (iterator): iterador na forma n; xm, xm-1, ... x1; p(xm, xm-1, ..., x1)
        L (int): número máximo de clientes simultâneos permitidos em sistema
        C (int): número de servidores ativos
        m (int): valor máximo possível de unidades de tempos de serviço
        lamb (float): parâmetro de taxa média de chegadas na fila
        k (float): parâmetro de forma da distribuição Gama
        theta (flaot): parâmetro de escala da distribuição Gama
        
    Yields:
        tuple: iterador na forma n2; wm, wm-1, ... w1; p(xm, xm-1, ..., x1)*v_r(t)*p(zm, zm-1, ..., z1)
    """
    vetores_n1 = ordenar_n1(remover_tempos_servico(vetores_n_x, C))

    for n1, vetores_y_a_px in vetores_n1:
        r_max = L - n1

        for r in range(r_max + 1):
            n2 = min(n1 + r, L)

            for ym_yi_y1, a, p_xm_xi_x1 in vetores_y_a_px:
                nnewst = min(n2, C) - a

                if nnewst == 0: continue
                vetores_z = gerar_vetores_z(nnewst, m)
                probs_z = gerar_probs_z(nnewst, vetores_z, m, k, theta)
                vetores_w = ym_yi_y1 + vetores_z
                probs_w = p_xm_xi_x1 * V(r, lamb) * probs_z

                for n2, wm_wi_w1, p_wm_wi_w1 in zip(*(np.repeat(n2, len(vetores_w)), vetores_w, probs_w)):
                    yield n2, list(wm_wi_w1), p_wm_wi_w1
                    
                    
def somar_probs_w(w):
    """
    Soma as probabilidades parciais para obter p(wm, wm-1, ..., w1)
    
    Args:
        w (iterator): iterador na forma n2; wm, wm-1, ... w1; p(xm, xm-1, ..., x1)*v_r(t)*p(zm, zm-1, ..., z1)
        
    Yields:
        tuple: iterador na forma n2; wm, wm-1, ... w1; p(wm, wm-1, ..., w1)
    """
    for k, g in it.groupby(sorted(w, key=itemgetter(0, 1)), key=lambda x: (x[0], x[1])):
        yield k[0], np.array(k[1]), sum([v[2] for v in g])
        
        
def ordenar_vetores_w(w_somado, m):
    """
    Agrupa e ordena os vetores (n2: wm, wm-1, ... w1) para melhorar a eficiência dos loops
    
    Args:
        w_somado (iterator): iterador na forma n2; wm, wm-1, ... w1; p(wm, wm-1, ..., w1)
        
    Yields:
        tuple: vetores agrupados utilizando n2 como chave
    """
    probs_w0 = 0
    
    for k, g in it.groupby(sorted(w_somado, key=itemgetter(0)), lambda x: x[0]):
        wm_wi_w1, p_wm_wi_w1 = zip(*[(v[1], v[2]) for v in g])
        probs_w0 += np.sum(p_wm_wi_w1)
        
        yield k, np.array(wm_wi_w1, dtype=int), np.array(p_wm_wi_w1, dtype=float)
        
    yield 0, np.array([np.zeros(m, dtype=int)], dtype=int), np.array([1 - probs_w0])

