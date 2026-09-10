# graph_rag.py
import os
import re
import json
import networkx as nx
from typing import List, Dict, Tuple
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import EMBEDDING_MODEL, LLM_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K, DATA_DIR, DB_DIR


class ChronosEngine:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        self.llm = ChatOllama(model=LLM_MODEL, temperature=0.3, num_predict=500)
        self.graph = nx.DiGraph()
        self.vectorstore = None
        self.documents = []
        
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(DB_DIR, exist_ok=True)

    def reset(self):
        self.documents = []
        self.graph = nx.DiGraph()
        self.vectorstore = None
        print("Reset completato.")

    def load_pdf(self, pdf_path: str):
        print("Carico " + pdf_path + "...")
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", "!", "?", ",", " "]
        )
        chunks = splitter.split_documents(pages)
        
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i
            chunk.metadata["source"] = os.path.basename(pdf_path)
        
        self.documents.extend(chunks)
        print("Creati " + str(len(chunks)) + " chunk.")
        return chunks

    def build_vectorstore(self):
        print("Costruisco il database vettoriale...")
        self.vectorstore = Chroma.from_documents(
            documents=self.documents,
            embedding=self.embeddings
        )
        print("Database pronto.")

    def extract_entities_fast(self, text: str) -> List[str]:
        entities = set()
        names = re.findall(r'\b[A-Z][a-z]+ (?:[A-Z][a-z]+ )*[A-Z][a-z]+\b', text)
        entities.update(names)
        dates = re.findall(r'\b(?:1[0-9]{3}|20[0-9]{2})\b', text)
        entities.update(dates)
        keywords = re.findall(r'\b[A-Z][A-Z\s]{2,}\b', text)
        entities.update(keywords)
        return list(entities)

    def build_knowledge_graph(self, max_chunks: int = 25):
        print("Costruisco il grafo di conoscenza (fast mode, max " + str(max_chunks) + " chunk)...")
        docs_to_process = self.documents[:max_chunks]
        
        for idx, doc in enumerate(docs_to_process):
            print("  Processo chunk " + str(idx+1) + "/" + str(len(docs_to_process)) + "...", end="\r")
            entities = self.extract_entities_fast(doc.page_content)
            chunk_id = doc.metadata.get("chunk_id", 0)
            
            for entity in entities:
                entity = entity.strip()
                if len(entity) > 2:
                    if not self.graph.has_node(entity):
                        self.graph.add_node(entity, type="entity", sources=[])
                    sources = self.graph.nodes[entity].get("sources", [])
                    if chunk_id not in sources:
                        sources.append(chunk_id)
                        self.graph.nodes[entity]["sources"] = sources
            
            for i, ent1 in enumerate(entities):
                for ent2 in entities[i+1:]:
                    ent1, ent2 = ent1.strip(), ent2.strip()
                    if ent1 != ent2 and len(ent1) > 2 and len(ent2) > 2:
                        if self.graph.has_edge(ent1, ent2):
                            self.graph[ent1][ent2]["weight"] += 1
                        else:
                            self.graph.add_edge(ent1, ent2, weight=1, relation="co-occurrence")
        
        print("\nGrafo pronto: " + str(self.graph.number_of_nodes()) + " nodi, " + str(self.graph.number_of_edges()) + " archi.")

    def _retrieve_context(self, question: str) -> Tuple[str, List[Document], List[str]]:
        if not self.vectorstore:
            return "", [], []
        
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": TOP_K})
        relevant_docs = retriever.invoke(question)
        
        question_lower = question.lower()
        related_entities = []
        for node in self.graph.nodes():
            if node.lower() in question_lower:
                neighbors = list(self.graph.neighbors(node))[:5]
                related_entities.extend(neighbors)
        
        additional_chunks = []
        for entity in related_entities:
            if entity in self.graph.nodes:
                chunk_ids = self.graph.nodes[entity].get("sources", [])[:2]
                for cid in chunk_ids:
                    for doc in self.documents:
                        if doc.metadata.get("chunk_id") == cid:
                            additional_chunks.append(doc)
        
        all_docs = relevant_docs + additional_chunks
        seen = set()
        unique_docs = []
        for d in all_docs:
            key = (d.metadata.get("chunk_id"), d.metadata.get("source"))
            if key not in seen:
                seen.add(key)
                unique_docs.append(d)
        
        context = "\n\n---\n\n".join([
            "[Fonte: " + d.metadata.get("source", "Sconosciuta") + ", Chunk " + str(d.metadata.get("chunk_id", "?")) + "]\n" + d.page_content
            for d in unique_docs[:7]
        ])
        
        return context, unique_docs, related_entities

    def query(self, question: str, mode: str = "tutor") -> Dict:
        context, unique_docs, related_entities = self._retrieve_context(question)
        
        if not context:
            return {"risposta": "Nessun documento caricato.", "fonti": [], "entita_collegate": [], "raw_docs": []}
        
        if mode == "socratic":
            prompt = (
                "Sei Socrate, il filosofo greco, ma applicato alla storia.\n"
                "Il tuo obiettivo NON e dare la risposta. Devi far ragionare lo studente con domande mirate.\n\n"
                "REGOLE ASSOLUTE:\n"
                "1. NON dare mai la risposta diretta alla domanda\n"
                "2. Fai 2-4 domande successive che guidino lo studente a scoprire la verita da solo\n"
                "3. Se citi un fatto specifico, indica la fonte [Fonte: nome, Chunk X]\n"
                "4. Fai una domanda finale aperta che inviti a riflettere sul perche degli eventi\n"
                "5. Se lo studente sembra confuso, scomponi il problema in parti piu semplici\n"
                "6. Usa un tono rispettoso ma stimolante, come un maestro che crede nelle capacita dello studente\n\n"
                "DOCUMENTI:\n" + context + "\n\n"
                "DOMANDA DELLO STUDENTE: " + question + "\n\n"
                "RISPOSTA SOCRATICA:"
            )
        elif mode == "narrator":
            prompt = (
                "Sei un narratore storico esperto e un attore consumato. Il tuo compito e mettere l'utente o un personaggio storico nel cuore degli eventi descritti nei documenti.\n\n"
                "REGOLE:\n"
                "1. Racconta in PRIMA PERSONA (io, noi, mi trovavo, vedevo, sentivo)\n"
                "2. Usa un tono emotivo, descrittivo, cinematografico. Descrivi suoni, odori, paura, eccitazione, folla\n"
                "3. Usa SOLO dettagli presenti nei documenti. NON inventare fatti\n"
                "4. Se la domanda non specifica un personaggio, scegli il protagonista piu adatto in base al contesto\n"
                "5. Cita le fonti quando menzioni fatti specifici [Fonte: nome, Chunk X]\n"
                "6. Inizia sempre con una frase d'impatto che ti pone nella scena\n"
                "7. Max 400 parole, concentrate e intense\n\n"
                "DOCUMENTI:\n" + context + "\n\n"
                "DOMANDA: " + question + "\n\n"
                "SCENA STORICA (in prima persona):"
            )
        else:
            prompt = (
                "Sei un tutor di storia esperto. Rispondi alla domanda usando SOLO le informazioni fornite nei documenti qui sotto.\n"
                "Cita sempre le fonti tra parentesi quadre [Fonte: nome, Chunk X].\n"
                "Se l'informazione non e nei documenti, dici 'Non trovo questa informazione nei documenti caricati.'\n\n"
                "DOCUMENTI:\n" + context + "\n\n"
                "DOMANDA: " + question + "\n\n"
                "RISPOSTA:"
            )
        
        response = self.llm.invoke(prompt)
        
        return {
            "risposta": response.content,
            "fonti": [
                {
                    "source": d.metadata.get("source", "Sconosciuta"),
                    "chunk": d.metadata.get("chunk_id", "?"),
                    "preview": d.page_content[:200] + "..."
                }
                for d in unique_docs[:3]
            ],
            "entita_collegate": related_entities[:5],
            "raw_docs": unique_docs[:5]
        }

    def verify_response(self, response_text: str, context_docs: List[Document]) -> Dict:
        if not context_docs:
            return {"status": "non_verificabile", "dettagli": ["Nessun documento di riferimento."]}
        
        context = "\n\n---\n\n".join([d.page_content[:800] for d in context_docs[:5]])
        
        prompt = (
            "Sei un rigoroso fact-checker storico. Confronta la RISPOSTA con i DOCUMENTI.\n"
            "Per ogni affermazione FATTUALE specifica nella RISPOSTA (date, nomi, eventi, numeri), "
            "verifica se e esplicitamente presente nei DOCUMENTI.\n"
            "Ignora opinioni, interpretazioni, connessioni logiche o domande retoriche.\n"
            "Se una affermazione non e nei documenti, segnala come POSSIBILE ALLUCINAZIONE.\n\n"
            "Rispondi in questo formato esatto, senza altro testo:\n"
            "STATO: VERIFICATO\n"
            "oppure\n"
            "STATO: PROBLEMI\n"
            "DETTAGLI:\n"
            "- [affermazione non supportata]\n"
            "- [affermazione non supportata]\n"
            "...\n\n"
            "DOCUMENTI:\n" + context + "\n\n"
            "RISPOSTA DA VERIFICARE:\n" + response_text + "\n\n"
            "VERIFICA:"
        )
        
        try:
            res = self.llm.invoke(prompt)
            content = res.content.strip()
            
            if "STATO: VERIFICATO" in content:
                return {"status": "verificato", "dettagli": []}
            else:
                dettagli = []
                in_dettagli = False
                for line in content.splitlines():
                    if line.startswith("DETTAGLI:"):
                        in_dettagli = True
                        continue
                    if in_dettagli and line.strip().startswith("-"):
                        dettagli.append(line.strip()[1:].strip())
                return {"status": "problemi", "dettagli": dettagli if dettagli else ["Risposta contiene affermazioni non verificabili dai documenti."]}
        except Exception as e:
            return {"status": "errore", "dettagli": ["Errore durante la verifica: " + str(e)]}

    def get_graph_stats(self) -> Dict:
        if self.graph.number_of_nodes() == 0:
            return {"nodi": 0, "archi": 0, "entita_top": []}
        
        degrees = dict(self.graph.degree())
        top_entities = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "nodi": self.graph.number_of_nodes(),
            "archi": self.graph.number_of_edges(),
            "entita_top": [{"entita": e, "connessioni": d} for e, d in top_entities]
        }

    def extract_timeline(self) -> List[Dict]:
        if not self.documents:
            return []
        
        events = []
        seen_keys = set()
        
        for doc in self.documents:
            text = doc.page_content.replace('\n', ' ')
            
            for match in re.finditer(r'\b(1[0-9]{3}|20[0-9]{2})\b', text):
                year = match.group(1)
                pos = match.start()
                
                sentence_start = max(0, pos - 120)
                for punct in ['. ', '! ', '? ']:
                    p = text.rfind(punct, sentence_start, pos)
                    if p != -1:
                        sentence_start = p + 2
                
                sentence_end = min(len(text), pos + 180)
                for punct in ['. ', '! ', '? ']:
                    p = text.find(punct, pos, sentence_end)
                    if p != -1:
                        sentence_end = p + 1
                
                sentence = text[sentence_start:sentence_end].strip()
                
                if len(sentence) < 15:
                    continue
                
                key = year + "|" + sentence[:50]
                if key not in seen_keys:
                    seen_keys.add(key)
                    events.append({
                        "anno": year,
                        "evento": sentence,
                        "descrizione": ""
                    })
        
        if not events:
            return []
        
        events.sort(key=lambda x: int(x['anno']))
        
        grouped = []
        current_year = None
        current_texts = []
        
        for ev in events:
            if ev['anno'] != current_year:
                if current_year and current_texts:
                    grouped.append({
                        "anno": current_year,
                        "evento": " / ".join(current_texts[:2]),
                        "descrizione": ""
                    })
                current_year = ev['anno']
                current_texts = [ev['evento']]
            else:
                current_texts.append(ev['evento'])
        
        if current_year and current_texts:
            grouped.append({
                "anno": current_year,
                "evento": " / ".join(current_texts[:2]),
                "descrizione": ""
            })
        
        return grouped