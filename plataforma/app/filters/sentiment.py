from .base import BaseFilter
from ..nlp.models.bertimbau_sentiment import BertimbauSentiment
from typing import Dict, Any
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class SentimentFilter(BaseFilter):
    """
    Filtro para análise de sentimento usando modelo BERTimbau especializado.
    """
    
    def __init__(self, model_path: str = None):
        super().__init__(
            name="Sentimento",
            description="Filtra por sentimento usando modelo BERTimbau especializado",
            default_enabled=True
        )
        
        # --- LÓGICA AUTOMÁTICA PARA ENCONTRAR O MODELO TREINADO ---
        if model_path is None:
            try:
                # Caminho base onde os modelos são salvos
                current_dir = Path(__file__).parent
                models_dir = current_dir.parent / 'nlp' / 'models' / 'trained'
                
                # Procura pastas que começam com 'AS_' (Análise de Sentimentos)
                if models_dir.exists():
                    model_paths = list(models_dir.glob('AS_*'))
                    if model_paths:
                        # Pega o mais recente (pela data de modificação)
                        latest_model = max(model_paths, key=lambda p: p.stat().st_mtime)
                        model_path = str(latest_model)
                        logger.info(f"Modelo de sentimento encontrado automaticamente: {model_path}")
                    else:
                        logger.warning("Nenhum modelo treinado 'AS_*' encontrado na pasta trained.")
                else:
                    logger.warning(f"Diretório de modelos não encontrado: {models_dir}")
            except Exception as e:
                logger.error(f"Erro ao tentar localizar modelo automaticamente: {e}")

        # ----------------------------------------------------------

        try:
            self.model = BertimbauSentiment(model_path=model_path)
            logger.info(f"Modelo de sentimentos carregado com sucesso de: {model_path}")
        except Exception as e:
            logger.error(f"Erro ao carregar modelo de sentimentos: {e}")
            self.model = None
        
    def process(self, video_data: Dict[str, Any]) -> float:
        """
        Processa o vídeo e retorna uma pontuação entre 0 e 1.
        O score é baseado na análise de sentimento do modelo especializado.
        """
        if self.model is None:
            logger.warning("Modelo não disponível, retornando score neutro")
            return 0.5
            
        # Junta título, descrição e transcrição para análise completa
        text = f"{video_data.get('title', '')} {video_data.get('description', '')} {video_data.get('transcript', '')}"
        
        if not text.strip():
            return 0.5  # Neutro quando não há texto
            
        try:
            # Usa o modelo especializado para análise
            result = self.model.predict_sentiment(text, return_probabilities=True)
            
            # Converte a classe predita em score (0-1)
            # 0=Negativo, 1=Neutro, 2=Positivo
            predicted_class = result['predicted_class']
            confidence = result['confidence']
            
            # Equação (3) do artigo: A = P(Positivo) + 0,5*P(Neutro), a leitura
            # graduada de adequação. Governa a faixa segura da Eq. (2) e é também o
            # percentual que a interface exibe.
            adequacy = self._adequacy(result, predicted_class, confidence)

            # Equação (2) do artigo: RISCO DOMINA TOM. Duas faixas disjuntas —
            # Negativo em [0,10; 0,233] e não-Negativo em [0,70; 0,85]. A distância
            # entre elas (0,467) é três vezes a variação interna (0,15): a gradação
            # por tom ordena entre pares sem risco, mas nunca coloca conteúdo sem
            # risco abaixo de conteúdo com risco. O teto é 0,85 e não 1,00 porque
            # não rebaixar não é atestar adequação — as demais dimensões de
            # inadequação não foram verificadas por este filtro.
            if predicted_class == 0:  # Negativo: a confiança gradua o rebaixamento
                score = 0.10 + (0.20 * (1 - confidence))
            else:  # Neutro ou Positivo: o tom ordena DENTRO da faixa segura
                score = 0.70 + (0.15 * adequacy)

            # A CLASSE viaja separada do escore, e é ela que colore o indicador na
            # interface. O escore ordena; a cor diz o que o item é.
            video_data["sentiment_class"] = ("negative", "neutral", "positive")[predicted_class]
            video_data["sentiment_adequacy"] = adequacy

            return min(max(score, 0.0), 1.0)  # Garante que está entre 0 e 1
            
        except Exception as e:
            logger.error(f"Erro ao processar sentimento: {e}")
            return 0.5  # Retorna neutro em caso de erro

    @staticmethod
    def _adequacy(result: Dict[str, Any], predicted_class: int, confidence: float) -> float:
        """Equação (3): A = P(Positivo) + 0,5 * P(Neutro).

        As probabilidades vêm num dict chaveado pelo nome da classe, construído com
        enumerate() sobre a saída do softmax — a ordem de inserção é, portanto, a
        ordem dos índices (0=Negativo, 1=Neutro, 2=Positivo). Usar a ordem em vez do
        nome mantém o cálculo válido se os rótulos do task_config mudarem.

        Sem as probabilidades, resta estimar pela confiança da classe predita, que é
        o que a classe base sempre devolve: reparte o restante igualmente entre as
        outras duas. É aproximação, e só entra em cenário degradado.
        """
        probabilidades = list((result.get("probabilities") or {}).values())
        if len(probabilidades) == 3:
            return float(probabilidades[2] + 0.5 * probabilidades[1])

        resto = (1.0 - confidence) / 2.0
        p = [resto, resto, resto]
        p[predicted_class] = confidence
        return float(p[2] + 0.5 * p[1])


    def get_filter_info(self) -> Dict[str, Any]:
        """
        Retorna informações sobre o filtro de sentimento.
        """
        return {
            "name": self.name,
            "description": self.description,
            "enabled": self.enabled,
            "weight": self.weight,
            "model_info": "BERTimbau Fine-tuned (AS)" if self.model else "Modelo não carregado",
            "options": {
                "sentiment_preference": {
                    "type": "slider",
                    "min": 0,
                    "max": 100,
                    "default": 50,
                    "description": "Preferência de sentimento (0=negativo, 50=neutro, 100=positivo)"
                }
            }
        }