# Como contribuir com o DroneSwarm2D

🇺🇸 [Read in English](CONTRIBUTING.md)

Obrigado pelo interesse! Relatos de bugs, ideias, novos comportamentos, documentação e código são todos bem-vindos.

## Preparando o ambiente

```bash
git clone https://github.com/lucasll37/DroneSwarm2D-lib.git
cd DroneSwarm2D-lib
python -m venv .venv && source .venv/bin/activate   # ou: make create_env (conda)
pip install -e ".[dev]"
```

A biblioteca Cairo precisa estar instalada (veja o [README](README.pt-BR.md#instalação)).

## Comandos do dia a dia

| Comando | O que faz |
|---|---|
| `make test` | Executa a suíte de testes em modo headless (cerca de 5 s) |
| `make coverage` | Testes com relatório de cobertura |
| `make lint` / `make format` | Verifica / corrige automaticamente o código com o ruff |
| `make demo ARGS="--windowed"` | Executa o exemplo |
| `make benchmark` | Mede a velocidade da simulação |

Os testes e o benchmark usam o driver de vídeo dummy do SDL, então rodam em máquinas sem display.

## Diretrizes

- **Preserve a retrocompatibilidade.** Os usuários rodam experimentos com seus próprios scripts. Não renomeie nem remova nomes públicos: adicione o nome novo e mantenha o antigo como alias (veja [Nomes obsoletos](docs/pt-BR/api.md#nomes-obsoletos)).
- **Mantenha as execuções reprodutíveis.** Use os geradores com semente (`FriendDrone.class_rng`, `EnemyDrone.class_rng`, `env.rng`) em vez dos globais `random` / `numpy.random`, e nunca use o relógio do sistema na lógica da simulação.
- **Meça antes e depois de mexer no desempenho.** Use `make benchmark` e confira se os resultados não mudaram (a mesma semente deve produzir o mesmo traço de execução).
- **Testes.** Toda correção de bug ganha um teste de regressão; toda funcionalidade nova ganha testes. Os testes não devem abrir janelas nem depender de temporização.
- **Estilo.**
  - O código passa no `ruff check`; as linhas têm no máximo 120 caracteres.
  - As docstrings seguem o estilo Google, em inglês.
  - As docstrings da API pública terminam com um breve resumo `PT-BR:`.
- **Documentação.** A documentação existe em inglês (`docs/en`, `README.md`) e em português brasileiro (`docs/pt-BR`, `*.pt-BR.md`). Atualize os dois idiomas no mesmo pull request.
- **Changelog.** Adicione uma linha ao `CHANGELOG.md` na seção *Unreleased*. Deixe isso explícito quando uma alteração mudar os resultados da simulação.

## Pull requests

1. Crie uma branch a partir de `main`.
2. Faça sua alteração com testes e documentação.
3. Execute `make lint test`.
4. Abra o pull request e descreva *o que* mudou e *por quê*. Se a alteração afetar os resultados, inclua os números de antes e depois.
