# -*- coding: utf-8 -*-
# =============================
# Simulação de Polarização Social e Retorno ao Pensamento Crítico
# Baseado em dinâmica de opinião, redes sociais e algoritmos
# =============================

# Importa bibliotecas para matemática, redes, visualização e estrutura de dados
from __future__ import annotations  # Permite referenciar tipos antes de serem definidos (útil para type hints)
import numpy as np                  # Biblioteca para cálculos numéricos (arrays, matrizes)
import math                         # Funções matemáticas (ex: logística)
import matplotlib.pyplot as plt     # Biblioteca para gráficos
from dataclasses import dataclass   # Cria classes com menos código (útil para configurações)

# =============================
# Utilidades de rede (Small-World)
# =============================

def watts_strogatz_graph(n: int, k: int, beta: float, rng: np.random.Generator):
    """
    Gera uma rede de Watts-Strogatz (Small-World).
    
    Exemplo:
    - n = 100 (100 pessoas)
    - k = 4 (cada pessoa conectada a 4 vizinhos)
    - beta = 0.1 (10% de chance de re-conectar aleatoriamente)
    
    Isso simula uma sociedade com conexões locais e algumas longas (como redes sociais).
    """
    # Garante que k seja par (necessário para rede anel)
    k = k if k % 2 == 0 else k + 1
    if k > n - 1: k = n - 1 if n > 1 else 0
    if k < 2 and n > 1: k = 2

    # Cria matriz de adjacência (0 = sem conexão, 1 = conectado)
    A = np.zeros((n, n), dtype=np.uint8)

    # Cria rede anel (cada nó conectado a k/2 vizinhos de cada lado)
    for i in range(n):
        for j in range(1, k//2 + 1):
            A[i, (i+j) % n] = 1  # Conecta à direita
            A[i, (i-j) % n] = 1  # Conecta à esquerda

    # Garante simetria (se A[i,j] = 1, então A[j,i] = 1)
    A = np.maximum(A, A.T)

    # Re-conecta arestas aleatoriamente com probabilidade beta
    for i in range(n):
        for j in range(1, k//2 + 1):
            old = (i + j) % n
            if i >= old:  # Evita processar aresta 2x
                continue
            if rng.random() < beta:
                A[i, old] = 0  # Remove antiga conexão
                A[old, i] = 0
                
                # Escolhe novo destino aleatório (sem duplicar aresta)
                candidates = np.where(A[i] == 0)[0]  # Nós não conectados
                candidates = candidates[candidates != i]  # Sem auto-conexão
                
                if len(candidates) > 0:
                    new = rng.choice(candidates)
                    A[i, new] = 1
                    A[new, i] = 1

    return A

# =============================
# Parâmetros do modelo
# =============================

@dataclass
class Params:
    """
    Configurações da simulação. Pense como os "controles" do seu experimento.
    
    Exemplo:
    - N = 400 → 400 pessoas simuladas
    - ALG_BIAS = 0.75 → algoritmo prioriza 75% conteúdo extremo
    - WAKE_THRESHOLD = 0.35 → quando polarização > 0.35, pessoas começam a "acordar"
    """
    N: int = 400                    # Número de agentes (pessoas)
    STEPS: int = 120                # Dias da simulação
    ALG_BIAS: float = 0.75          # Viés do algoritmo: 0 = neutro, 1 = só extremos
    ALG_GAMMA: float = 1.5          # Expoente de "extremidade" no feed (quanto mais extremo, mais viral)
    FEED_INERTIA: float = 0.2       # Quanto a opinião muda com base no feed (0.2 = pouca mudança)
    HK_EPS: float = 0.35            # Tolerância para aceitar opinião de vizinho (Confiança Limitada)
    DEGROOT_W: float = 0.3          # Peso da influência social para leitores
    PASSIVE_PULL: float = 0.03      # Força que puxa passivos de volta ao centro
    STUBBORN_FRAC: float = 0.05     # 5% são "teimosos" (não mudam de opinião)
    WAKE_THRESHOLD: float = 0.35    # Limiar de polarização para despertar
    WAKE_MAX_P: float = 0.08        # Máximo de pessoas que acordam por dia (8%)
    WAKE_SHARPNESS: float = 14.0    # Quão "abrupto" é o despertar (função logística)
    PEER_CONTAGION: float = 0.30    # Chance de virar leitor por influência de vizinhos
    REWIRE_P: float = 0.03          # Chance de re-conectar com semelhantes (bolhas)
    REWIRE_BONUS: float = 0.7       # Quanto mais semelhantes, mais conexões
    SHOCK_FREQ: int = 20            # A cada 20 dias, eventos externos (notícias, crises)
    SHOCK_STD: float = 0.1          # Intensidade dos choques (ruído)
    GRAPH_K: int = 8                # Grau médio inicial da rede
    GRAPH_BETA: float = 0.1         # Aleatoriedade da rede (0 = estrutura local, 1 = aleatório)
    SEED: int = 42                  # Semente para reprodutibilidade

# =============================
# Núcleo do modelo
# =============================

class Society:
    """
    Classe que representa a sociedade simulada.
    
    Exemplo:
    - Um agente pode ser:
      - 0: Passivo (influenciado por algoritmos)
      - 1: Leitor (busca conteúdo crítico)
      - 2: Teimoso (não muda de opinião)
    """
    def __init__(self, p: Params):
        self.p = p
        self.rng = np.random.default_rng(p.SEED)  # Gerador de números aleatórios
        self.A = watts_strogatz_graph(p.N, p.GRAPH_K, p.GRAPH_BETA, self.rng)  # Rede social

        # Estados dos agentes: 0 = passivo, 1 = leitor, 2 = teimoso
        self.state = np.zeros(p.N, dtype=np.int8)

        # Inicializa opiniões aleatórias entre -1 (esquerda) e +1 (direita)
        self.op = self.rng.uniform(-1, 1, p.N)

        # Define alguns como teimosos (opinião fixa)
        num_stubborn = int(p.N * p.STUBBORN_FRAC)
        stubborn_idx = self.rng.choice(p.N, size=num_stubborn, replace=False)
        self.state[stubborn_idx] = 2
        stubborn_signs = self.rng.choice([-1.0, 1.0], size=num_stubborn)
        self.op[stubborn_idx] = stubborn_signs

        # Históricos
        self.pol_hist = []      # Polarização ao longo do tempo
        self.read_hist = []     # Fração de leitores
        self.echo_hist = []     # Índice de eco-câmara
        self.t = 0              # Tempo atual

    # ---------- Métricas ----------

    def polarization(self):
        """
        Calcula a polarização como o desvio padrão das opiniões.
        
        Exemplo:
        - Opiniões: [0.1, 0.2, 0.15] → desvio = 0.05 (pouco polarizado)
        - Opiniões: [-0.9, 0.8, -0.7] → desvio = 0.7 (muito polarizado)
        """
        return float(np.std(self.op))

    def readers_frac(self):
        """Fração de agentes que são leitores (estado = 1)."""
        return float(np.mean(self.state == 1))

    def echo_index(self):
        """
        Mede o quão homogênea é a rede (eco-câmara).
        
        Exemplo:
        - Se 80% das conexões são entre pessoas com mesma opinião → eco_index = 0.8
        - Se metade é diversa, metade homogênea → eco_index = 0.5
        """
        i_idx, j_idx = np.where(np.triu(self.A, 1) == 1)  # Pares conectados (sem duplicar)
        if len(i_idx) == 0:
            return 0.0
        same = np.sign(self.op[i_idx]) * np.sign(self.op[j_idx]) >= 0  # Mesmo sinal?
        return float(np.mean(same))

    # ---------- Dinâmicas ----------

    def algorithmic_feed(self, i):
        """
        Simula o que um agente passivo vê no feed com base em algoritmo.
        
        Exemplo:
        - Agente i tem opinião 0.3 (moderado)
        - Feed prioriza conteúdo extremo (±0.8, ±0.9)
        - Algoritmo mostra opiniões parecidas com i (similaridade)
        """
        p = self.p
        opinions = self.op

        # Peso de relevância: combina "extremidade" e "similaridade"
        extremeness = np.abs(opinions) ** p.ALG_GAMMA  # Quanto mais extremo, mais viral
        sim = 0.5 * (1 + np.sign(opinions) * np.sign(opinions[i]))  # 1 se mesmo sinal, 0 se oposto

        # Combinação de viés por extremo e similaridade
        score = (1 - p.ALG_BIAS) * extremeness + p.ALG_BIAS * (sim + extremeness)

        # Amostra aleatória ponderada (quem tem score alto aparece mais)
        k = max(5, int(0.1 * p.N))  # Tamanho da amostra (10% da população)
        probs = score / (score.sum() + 1e-9)
        probs[i] = 0.0  # Evita auto-referência
        probs /= probs.sum() + 1e-12

        idx = self.rng.choice(p.N, size=k, replace=False, p=probs)

        # Confiança limitada: só aceita opiniões "próximas"
        mask = np.abs(opinions[idx] - opinions[i]) <= p.HK_EPS

        if mask.any():
            return float(opinions[idx][mask].mean())
        else:
            return float(opinions[i])

    def social_influence(self, i):
        """
        Influência social baseada em vizinhos aceitos (Confiança Limitada - HK).
        
        Exemplo:
        - Agente i conectado a 5 vizinhos
        - Apenas 2 têm opiniões "próximas" (diferença < HK_EPS)
        - Média das opiniões desses 2 influencia i
        """
        p = self.p
        neighbors = np.where(self.A[i] == 1)[0]  # Vizinhos conectados

        if len(neighbors) == 0:
            return float(self.op[i])

        close = np.abs(self.op[neighbors] - self.op[i]) <= p.HK_EPS  # Aceitos?

        if not np.any(close):
            return float(self.op[i])

        return float(self.op[neighbors][close].mean())

    def step_updates(self):
        """
        Atualiza o estado da sociedade em um passo de tempo.
        """
        p = self.p
        new_op = self.op.copy()

        # Choques exógenos (eventos externos como crises)
        if p.SHOCK_FREQ and (self.t % p.SHOCK_FREQ == 0) and (self.t > 0):
            shock = self.rng.normal(0, p.SHOCK_STD, size=p.N)
            new_op = np.clip(new_op + shock, -1, 1)

        # Atualização de opiniões
        for i in range(p.N):
            if self.state[i] == 2:  # Teimoso: não muda
                continue

            current_op = self.op[i]

            if self.state[i] == 1:  # Leitor: influenciado por vizinhos
                nb_mean = self.social_influence(i)
                new_op[i] = (1 - p.DEGROOT_W) * current_op + p.DEGROOT_W * nb_mean

            else:  # Passivo: influenciado por algoritmo
                feed = self.algorithmic_feed(i)
                new_op[i] = (1 - p.FEED_INERTIA) * current_op + p.FEED_INERTIA * feed
                # Puxão ao centro
                new_op[i] = (1 - p.PASSIVE_PULL) * new_op[i] + p.PASSIVE_PULL * 0.0

        self.op = np.clip(new_op, -1, 1)

        # --- Despertar e contágio ---
        pol = self.polarization()
        wake_p = self._wake_probability(pol)

        for i in range(p.N):
            if self.state[i] != 0:  # Apenas passivos podem acordar
                continue

            # Despertar global (baseado em polarização)
            if self.rng.random() < wake_p:
                self.state[i] = 1
                continue

            # Contágio local (vizinhos leitores influenciam)
            neighbors = np.where(self.A[i] == 1)[0]
            if len(neighbors) > 0:
                frac_readers = np.mean(self.state[neighbors] == 1)
                if self.rng.random() < p.PEER_CONTAGION * frac_readers:
                    self.state[i] = 1

        # --- Rewire homofílico (bolhas de filtro) ---
        self._homophily_rewire()

        # Registra métricas
        self.pol_hist.append(pol)
        self.read_hist.append(self.readers_frac())
        self.echo_hist.append(self.echo_index())

    def _wake_probability(self, pol):
        """
        Calcula a chance de alguém despertar com base na polarização.
        
        Exemplo:
        - Polarização = 0.4, limiar = 0.35 → chance aumenta
        - Função logística: suave, mas cresce rápido após limiar
        """
        p = self.p
        x = p.WAKE_SHARPNESS * (pol - p.WAKE_THRESHOLD)
        return p.WAKE_MAX_P / (1 + math.exp(-x))

    def _homophily_rewire(self):
        """
        Reconecta arestas com base em similaridade (formação de bolhas).
        
        Exemplo:
        - Agente i tem opinião 0.8
        - Ele tende a se conectar com outros 0.7, 0.9
        - Desconecta de -0.8, -0.9 (opostos)
        """
        p = self.p
        if p.REWIRE_P <= 0:
            return

        m = max(1, int(p.REWIRE_P * p.N))
        nodes = self.rng.choice(p.N, size=m, replace=False)

        for i in nodes:
            bias = p.REWIRE_BONUS * (1.0 if self.state[i] == 0 else 0.4)  # Passivos têm mais homofilia

            nb = np.where(self.A[i] == 1)[0]
            if len(nb) < 2:
                continue

            # Quebra aresta com menor similaridade
            similarity = 1 - np.abs(self.op[nb] - self.op[i]) / 2
            break_probs = (1 - similarity)
            break_probs = np.maximum(break_probs, 1e-9)
            break_probs /= break_probs.sum()
            j = self.rng.choice(nb, p=break_probs)
            self.A[i, j] = self.A[j, i] = 0

            # Conecta a alguém mais semelhante
            candidates = np.where(self.A[i] == 0)[0]
            candidates = candidates[candidates != i]
            if len(candidates) == 0:
                self.A[i, j] = self.A[j, i] = 1  # Reverte se não há para onde ligar
                continue

            sim = 1 - np.abs(self.op[candidates] - self.op[i]) / 2
            uniform_probs = np.ones_like(sim) / len(sim)
            sim_norm = sim / (sim.sum() + 1e-9)
            probs = (1 - bias) * uniform_probs + bias * sim_norm
            probs = np.maximum(probs, 0)
            probs /= probs.sum()
            new = self.rng.choice(candidates, p=probs)
            self.A[i, new] = self.A[new, i] = 1

    def run(self):
        """
        Executa a simulação por p.STEPS passos e retorna métricas.
        """
        self.pol_hist.clear(); self.read_hist.clear(); self.echo_hist.clear()
        for t in range(self.p.STEPS):
            self.t = t
            self.step_updates()
        return {
            'polarization': np.array(self.pol_hist),
            'readers': np.array(self.read_hist),
            'echo': np.array(self.echo_hist),
            'final_opinions': self.op.copy(),
            'states': self.state.copy()
        }

# =============================
# Visualização (gráficos separados)
# =============================

def plot_metrics(res, p: Params):
    """
    Plota os gráficos de polarização, leitores e eco-câmara.
    """
    pl = res['polarization']
    rd = res['readers']
    echo = res['echo']

    # Gráfico 1: Polarização
    plt.figure(figsize=(9, 5))
    plt.plot(pl, linewidth=2, label='Polarização (DP)')
    plt.axhline(y=p.WAKE_THRESHOLD, color='r', linestyle='--', alpha=0.6, label='Limiar despertar')
    plt.title('Polarização (Desvio Padrão) ao longo do tempo')
    plt.xlabel('Dias'); plt.ylabel('Desvio Padrão'); plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.show()

    # Gráfico 2: Leitores
    plt.figure(figsize=(9, 5))
    plt.plot(rd, linewidth=2, color='g')
    plt.title('Fração de Leitores (Literacia/Consumo Aprofundado)')
    plt.xlabel('Dias'); plt.ylabel('Fração'); plt.ylim(0, max(1, rd.max()*1.1)); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.show()

    # Gráfico 3: Eco-câmara
    plt.figure(figsize=(9, 5))
    plt.plot(echo, linewidth=2, color='b')
    plt.title('Índice de Eco-Câmara (Fração de Laços Homofílicos)')
    plt.xlabel('Dias'); plt.ylabel('0 = diverso, 1 = homogêneo'); plt.ylim(0, 1); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.show()

# =============================
# Execução Principal
# =============================

if __name__ == "__main__":
    p = Params()
    print("--- PARÂMETROS BASE ---")
    print(f"Nós: {p.N}, Passos: {p.STEPS}, Teimosos: {p.STUBBORN_FRAC:.1%}, EPS (HK): {p.HK_EPS:.2f}")

    soc = Society(p)
    res = soc.run()

    print("\n=== RESULTADOS FINAIS ===")
    print(f"Polarização inicial ~ {res['polarization'][0]:.3f}")
    print(f"Polarização máxima  ~ {res['polarization'].max():.3f}")
    print(f"Polarização final   ~ {res['polarization'][-1]:.3f}")
    print(f"Leitores finais     ~ {res['readers'][-1]:.1%}")
    print(f"Eco-câmara final    ~ {res['echo'][-1]:.2f}")

    plot_metrics(res, p)

    # Histograma de opiniões finais
    plt.figure(figsize=(9, 5))
    passives = res['final_opinions'][res['states'] == 0]
    readers = res['final_opinions'][res['states'] == 1]
    stubborn = res['final_opinions'][res['states'] == 2]

    plt.hist(passives, bins=np.linspace(-1, 1, 21), alpha=0.6, label='Passivos', color='lightgray')
    plt.hist(readers, bins=np.linspace(-1, 1, 21), alpha=0.7, label='Leitores', color='skyblue')
    plt.hist(stubborn, bins=np.linspace(-1, 1, 21), alpha=0.9, label='Teimosos (Zealots)', color='red')

    plt.title('Distribuição Final de Opiniões por Tipo de Agente')
    plt.xlabel('Opinião (-1 a 1)'); plt.ylabel('Contagem')
    plt.legend(); plt.grid(axis='y', alpha=0.3); plt.tight_layout()
    plt.show()
