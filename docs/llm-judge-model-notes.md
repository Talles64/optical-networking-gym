# LLM Judge — Model Behavior Notes

Testes realizados em 2026-03-21 com o endpoint Ollama local (`http://localhost:11434`),
usando a task de julgamento de heurísticas (structured JSON output com schema flat).

---

## Parâmetro `think`

O Ollama aceita `think` como `bool` ou `string` no corpo da requisição (`/api/chat`).
Suporte real ao controle de budget varia por modelo.

Valores suportados: `false | "low" | "medium" | "high" | true`

---

## qwen3.5:4b

Modelo de thinking. Com `think=False` desativa o raciocínio e responde rapidamente.
Os valores string (`"low"`, `"medium"`) não controlam o budget de forma confiável —
comportam-se como `true` com variação aleatória.

| think | tempo | thinking words | output words | notas |
|-------|-------|---------------|-------------|-------|
| `False` | ~1–5s | 0 | ~300 | recomendado para produção |
| `"low"` | ~60s | ~3900 | ~450 | imprevisível, não usar |
| `"medium"` | ~37s | ~2100 | ~430 | imprevisível, não usar |

**Recomendação:** usar `think=False`. O schema JSON guia o output sem precisar de raciocínio.

**Atenção:** o modelo não segue o JSON schema à risca — retorna `"signals"` em vez de
`"decisive_signals"`. O parser aceita ambos como alias.
Ocasionalmente retorna uma lista em vez de objeto (falha graciosamente, step ignorado).

---

## gpt-oss:20b

Modelo de thinking. `think=False` **não desativa** o raciocínio — o modelo pensa por padrão
independente do parâmetro. Os valores string funcionam como budgets reais.

| think | tempo | thinking words | output words | notas |
|-------|-------|---------------|-------------|-------|
| `False` | ~20s | ~425 | ~456 | ignora o parâmetro, pensa mesmo assim |
| `"low"` | ~1.4s | ~10 | ~80 | mais rápido, output mais curto |
| `"medium"` | ~8.6s | ~304 | ~410 | balanço ideal |

**Recomendação:** usar `think="low"` para throughput alto, `think="medium"` para qualidade.

---

## Configuração no script (`online_heuristic_judge.py`)

```python
# ── LLM Judge inline config (edit here to switch models) ─────────────────────
_OLLAMA_BASE_URL         = "http://localhost:11434"
_OLLAMA_MODEL            = "qwen3.5:4b"
_OLLAMA_TEMPERATURE      = 0.0
_OLLAMA_TIMEOUT_S        = 60.0
_OLLAMA_MAX_RETRIES      = 1
_OLLAMA_SKIP_EXPLANATION = True
_OLLAMA_THINK            = False   # False | "low" | "medium" | "high" | True  (support varies by model)
# ─────────────────────────────────────────────────────────────────────────────
```

### Combinações testadas e validadas

| model | think | status |
|-------|-------|--------|
| `qwen3.5:4b` | `False` | ✅ funciona, ~1–5s/step |
| `qwen3.5:4b` | `"low"` / `"medium"` | ⚠️ imprevisível, lento |
| `gpt-oss:20b` | `False` | ⚠️ pensa mesmo assim (~20s) |
| `gpt-oss:20b` | `"low"` | ✅ funciona, ~1.4s/step |
| `gpt-oss:20b` | `"medium"` | ✅ funciona, ~8.6s/step |

---

## Problemas conhecidos

- **langchain_ollama**: `ChatOllama` ignora o parâmetro `timeout` em v1.0.1 e usa `stream=True`
  por padrão, o que causa hang indefinido em modelos de thinking. Substituído por `ollama.Client` direto.
- **Timeout**: `ollama.Client(timeout=N)` funciona corretamente via httpx.
- **Schema compliance**: modelos pequenos às vezes retornam arrays em vez de objetos —
  o pipeline captura a exceção e registra o erro no JSONL sem interromper o episódio.
