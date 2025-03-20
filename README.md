# FLAME: Federated Learning for Attack Mitigation and Evasion

## Abstract
In today's interconnected cyber landscape, Distributed Denial of Service (DDoS) attacks represent a significant threat to the smooth functioning of online infrastructures. 
The nature of DDoS attacks, characterized by their distributed and dynamic nature, poses significant challenges for traditional centralized approaches to model training; however, the challenges of collaborative DDoS detection are compounded by stringent data privacy regulations, leaving mitigation efforts largely reliant on standalone and inflexible firewalls. Federated Learning (FL) represents a cutting-edge innovation in cybersecurity, presenting a revolutionary method for collectively training deep learning models without compromising sensitive data. Despite its promise, practical hurdles remain, particularly the reliance of most FL algorithms on centralized, server-side data for model evaluation—though some approaches avoid this centralized testing dependency.
This limitation hinders the applicability of FL, especially in scenarios involving zero-day attacks on clients. Our paper examines a key hypothesis: whether the aggregated information from multiple clients can be effectively utilized to develop a global model that is inherently more resilient to zero-day attacks compared to models trained solely on individual client data. To investigate this, we introduce a methodology wherein FL models are trained on established DDoS attacks and subsequently evaluated against entirely novel, unencountered attacks, simulating zero-day scenarios at the client level. To ensure that each client contributes effectively to the training process, we utilize Jensen-Shannon Divergence (JSD) to evaluate and filter client updates based on their alignment with the global model. Building on this, we implement a kernel density estimation-based aggregation method to effectively mitigate feature distribution bias—a common issue in DDoS detection within FL environments. This approach forms a core component of our proposed framework, FLAME, which is built using the distributed framework Flower to realistically simulate FL in a decentralized setting.

## Acknowledgement
- The authors thank the IBiSco project (Infrastructure for BIg data and Scientific COmputing), PON R\&I 2014-2020 under Call 424-2018 - Action II.1, for the support and use of the HPC Cluster.
  
- G.A.N.D.A.L.F. - Gan Approaches for Non-iiD Aiding Learning in Federations, CUP: E53D23008290006, PNRR - Missione 4 “Istruzione e Ricerca” - Componente C2 Investimento 1.1 “Fondo per il Programma Nazionale di Ricerca e Progetti di Rilevante Interesse Nazionale (PRIN)”.


### Note
Place the CICDDoS2019 data set in the folder '../FLAME/data_raw'.
