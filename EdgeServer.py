import time
import threading
import random
from queue import Empty
from blspy import PopSchemeMPL

class EdgeServer(threading.Thread):
    def __init__(self, id, n, is_corrupted, server_type, hash_ds, 
                 private_key, public_keys, proof, clusters, c_id, ch_id, 
                 cluster_heads, latency_matrix, ss_queue, sp_queues, ap_queues, 
                 lv_queues, gv_queues, dt1, dt2, dt3, t1s, t2s, l_times, 
                 g_times, l_verdicts, g_verdicts, data_hashing_delay, 
                 hash_signing_delay, proof_aggregation_delays, 
                 proof_verification_delays):
        threading.Thread.__init__(self)
        self.id = id
        self.n = n
        self.is_corrupted = is_corrupted
        self.server_type = server_type
        self.hash_d = hash_ds[id]
        self.hash_ds = hash_ds
        self.private_key = private_key
        self.public_key = public_keys[id]
        self.public_keys = public_keys
        self.proof = proof
        self.clusters = clusters
        self.cluster_heads = cluster_heads
        self.c_id = c_id
        self.ch_id = ch_id
        self.is_cluster_head = self.ch_id == self.id
        self.latency_matrix = latency_matrix
        self.ss_queue = ss_queue
        self.sp_queues = sp_queues
        self.ap_queues = ap_queues
        self.lv_queues = lv_queues
        self.gv_queues = gv_queues
        self.dt1 = dt1
        self.dt2 = dt2
        self.dt3 = dt3
        self.t1s = t1s
        self.t2s = t2s
        self.l_times = l_times
        self.g_times = g_times
        self.similar_proofs = {self.id: self.proof}
        self.distinct_proofs = {}
        self.similar_agg_proofs = {}
        self.distinct_agg_proofs = {}
        self.similar_proof_count = 1
        self.distinct_proof_count = 0
        self.data_hashing_delay = data_hashing_delay
        self.hash_signing_delay = hash_signing_delay
        self.proof_aggregation_delays = proof_aggregation_delays
        self.proof_verification_delays = proof_verification_delays
        self.l_verdicts = l_verdicts
        self.g_verdicts = g_verdicts

    def __str__(self):
        return f"EdgeServer {self.id}"
    
    def run(self):
        # Wait until all threads are initialized. This is done because Python
        # thread initialization takes much time.
        self.ss_queue.put(self.id)
        while self.ss_queue.qsize() < self.n:
            time.sleep(0.0001)
        
        # Set the start time of the thread to measure the elapsed time.
        self.t0 = time.time()

        # Start the thread as a cluster head or cluster member.
        if self.is_cluster_head:
            self.run_ch()
        else:
            self.run_cm()   

    ############################################################################
    # Cluster Member Methods
    ############################################################################

    def run_cm(self):
        # Send proofs to the cluster head.
        threading.Thread(target=self.send_proof_to_cluster_head).start()
        
        # Listen to local verdict until t2 (dt1+dt2).
        threading.Thread(target=self.listen_to_local_verdict).start()
        
        # Listen to global verdict from until t3 (dt1+dt2+dt3).
        threading.Thread(target=self.listen_to_global_verdict).start()

    def send_proof_to_cluster_head(self):
        # Add delay to simulate the data hashing and  hash signing process.
        time.sleep(self.data_hashing_delay + self.hash_signing_delay)
        
        # Simulate network latency, if latency < 0, the message is dropped
        # assuming the malicious/faulty behaviour of the edge server.
        latency = self.latency_matrix[self.id][self.ch_id]
        if latency >= 0:
            time.sleep(latency)
            self.sp_queues[self.ch_id].put({"id": self.id, "proof": self.proof})

    def listen_to_local_verdict(self):
        try:
            lv = self.lv_queues[self.id].get(timeout=self.dt1+self.dt2)
        except Empty:
            lv = {"agg_proof": None}
        if lv["agg_proof"]:
            pks = [self.public_keys[id] for id in lv["ids"]]
            time.sleep(random.uniform(
                self.proof_verification_delays[str(len(pks))]["min"],
                self.proof_verification_delays[str(len(pks))]["max"]))
            self.l_verdicts[self.id] = PopSchemeMPL.fast_aggregate_verify(
                pks, self.hash_d, lv["agg_proof"])
            if self.l_verdicts[self.id]:
                self.l_times[self.id] = time.time() - self.t0

    def listen_to_global_verdict(self):
        try:
            gv = self.gv_queues[self.id].get(timeout=self.dt1+self.dt2+self.dt3)
        except Empty:
            gv = {"agg_proof": None}
        if gv["agg_proof"]:
            pks = [self.public_keys[id] for id in gv["ids"]]
            time.sleep(random.uniform(
                self.proof_verification_delays[str(len(pks))]["min"],
                self.proof_verification_delays[str(len(pks))]["max"]))
            if self.hash_d == gv["hash_d"]:
                self.g_verdicts[self.id] = PopSchemeMPL.fast_aggregate_verify(
                    pks, self.hash_d, gv["agg_proof"])
            else:
                self.g_verdicts[self.id] = PopSchemeMPL.fast_aggregate_verify(
                    pks, gv["hash_d"], gv["agg_proof"])
            if self.g_verdicts[self.id]:
                self.g_times[self.id] = time.time() - self.t0
            else:
                # If cluster head provided proof cannot be verified, edge server
                # contacts app vendor as the fallback mechanism.
                # print("cluster head provided proof cannot be verified")
                self.g_times[self.id] = time.time() - self.t0 + 0.5
        else:
            # If cluster head is unable to provide a proof, then edge server
            # contacts app vendor as the fallback mechanism.
            # print("cluster head is unable to provide a proof")
            self.g_times[self.id] = time.time() - self.t0 + 0.5

    ############################################################################
    # Cluster Head Methods
    ############################################################################

    def run_ch(self):
        # Verify local proofs until t1 or majority is identified. Then, send 
        # aggregated proof to other cluster heads and cluster members.
        threading.Thread(
            target=self.verify_local_proofs_and_send_agg_proof).start()
        
        # Verify aggregated proofs from other cluster heads. Then, send the
        # global verdict to cluster members.
        threading.Thread(
            target=self.verify_agg_proofs_and_send_global_verdict).start()

    def verify_local_proofs_and_send_agg_proof(self):
        while (time.time() - self.t0) < self.dt1 and len(
            self.similar_proofs) <= len(self.clusters[self.c_id])/2:
            if not self.sp_queues[self.id].empty():
                sp = self.sp_queues[self.id].get()
                time.sleep(random.uniform(
                self.proof_verification_delays["1"]["min"],
                self.proof_verification_delays["1"]["max"]))
                ok = PopSchemeMPL.verify(self.public_keys[sp["id"]], 
                                         self.hash_d, sp["proof"])
                if ok:
                    self.similar_proofs[sp["id"]] = sp["proof"]
                else:
                    self.distinct_proofs[sp["id"]] = sp["proof"]
            time.sleep(0.0001)
        self.t1s[self.id] = time.time() - self.t0
        
        # If more than half of the proofs are similar, aggregate them.
        # Otherwise, create empty aggregated proof.
        if len(self.similar_proofs) > len(self.clusters[self.c_id])/2:
            time.sleep(random.uniform(
                self.proof_aggregation_delays[str(len(
                    self.similar_proofs))]["min"],
                self.proof_aggregation_delays[str(len(
                    self.similar_proofs))]["max"]))
            intra_agg_proof = PopSchemeMPL.aggregate(list(
                self.similar_proofs.values()))
            self.l_verdicts[self.id] = True
            self.l_times[self.id] = time.time() - self.t0
        else:
            intra_agg_proof = None
        ids = self.similar_proofs.keys() if intra_agg_proof else []

        # Send the agg_proof to other cluster members as local integrity.
        lv = {"ids": ids, "agg_proof": intra_agg_proof}
        for s_id in self.clusters[self.c_id]:
            if s_id == self.id:
                continue
            threading.Thread(target=self.send_local_verdict_to_cluster_members, 
                             args=(s_id, lv)).start()
        
        # Send the agg_proof to other cluster heads.
        ap = {"id": self.id, "proof": self.proof, "ids": ids, 
              "agg_proof": intra_agg_proof}
        for ch_id in self.cluster_heads.values():
            if ch_id == self.id:
                self.similar_agg_proofs[self.id] = ap
                self.similar_proof_count += len(ids)
                continue
            threading.Thread(target=self.send_agg_proof_to_cluster_heads, 
                             args=(ch_id, ap)).start()
            
    def send_local_verdict_to_cluster_members(self, s_id, lv):
        latency = self.latency_matrix[self.id][s_id]
        if latency >= 0:
            time.sleep(latency)
        self.lv_queues[s_id].put(lv)
        
    def send_agg_proof_to_cluster_heads(self, ch_id, ap):
        latency = self.latency_matrix[self.id][ch_id]
        if latency >= 0:
            time.sleep(latency)
        self.ap_queues[ch_id].put(ap)
        
    def verify_agg_proofs_and_send_global_verdict(self):
        while ((time.time() - self.t0) < self.dt1+self.dt2) and (
            self.similar_proof_count <= self.n/3):
            if not self.ap_queues[self.id].empty():
                ap = self.ap_queues[self.id].get()
                pks = [self.public_keys[id] for id in ap["ids"]]
                if ap["agg_proof"] and PopSchemeMPL.fast_aggregate_verify(
                    pks, self.hash_d, ap["agg_proof"]):
                    time.sleep(random.uniform(
                        self.proof_verification_delays[str(len(pks))]["min"],
                        self.proof_verification_delays[str(len(pks))]["max"]))
                    self.similar_agg_proofs[ap["id"]] = ap
                    self.similar_proof_count += len(pks)
                else:
                    self.distinct_agg_proofs[ap["id"]] = ap
                    self.distinct_proof_count += len(pks)
            time.sleep(0.0001)
        self.t2s[self.id] = time.time() - self.t0
        
        if self.similar_proof_count > self.n/3:
            time.sleep(random.uniform(
                self.proof_aggregation_delays[str(len(
                    self.similar_agg_proofs))]["min"],
                self.proof_aggregation_delays[str(len(
                    self.similar_agg_proofs))]["max"]))
            inter_agg_proof = PopSchemeMPL.aggregate(
                [self.similar_agg_proofs[ch_id]["proof"] 
                 for ch_id in self.similar_agg_proofs.keys()])
            ids = self.similar_agg_proofs.keys()
            hash_d = self.hash_d
        else:
            # Simulate network latency to get the hash_d values from other
            # cluster heads. Thus, sleeping for the maximum latency of the
            # cluster heads.
            max_latency = max([self.latency_matrix[self.id][id] 
                               for id in self.cluster_heads.values()])
            if max_latency >=0:
                time.sleep(max_latency)

            # Retrieve the hash_d values from other cluster heads. Then, get the
            # mode of hash_d values.
            hash_ds = [self.hash_ds[ch_id] 
                       for ch_id in self.cluster_heads.values()]
            hash_d = max(set(hash_ds), key=hash_ds.count)

            # Verify the aggregated proofs of the cluster heads using hash_d.
            ids = []
            sp_count = 0
            aps = list(self.similar_agg_proofs.values()) + list(
                self.distinct_agg_proofs.values())
            for ap in aps:
                pks = [self.public_keys[id] for id in ap["ids"]]
                if ap["agg_proof"] and PopSchemeMPL.fast_aggregate_verify(
                    pks, hash_d, ap["agg_proof"]):
                    time.sleep(random.uniform(
                        self.proof_verification_delays[str(len(pks))]["min"],
                        self.proof_verification_delays[str(len(pks))]["max"]))
                    ids.append(ap["id"])
                    sp_count += len(pks)
            if sp_count > self.n/3:
                time.sleep(random.uniform(
                    self.proof_aggregation_delays[str(len(
                        self.distinct_agg_proofs))]["min"],
                    self.proof_aggregation_delays[str(len(
                        self.distinct_agg_proofs))]["max"]))
                inter_agg_proof = PopSchemeMPL.aggregate(
                    [self.distinct_agg_proofs[id]["proof"] for id in ids])
            else:
                inter_agg_proof = None
        
        # Send the agg_proof to cluster members as global verdict.
        gv = {"ids": ids, "agg_proof": inter_agg_proof, "hash_d": hash_d}
        for s_id in self.clusters[self.c_id]:
            threading.Thread(target=self.send_global_verdict_to_cluster_members, 
                             args=(s_id, gv)).start()
        
        # Set global verdict.
        if inter_agg_proof:
            self.g_verdicts[self.id] = True
            self.g_times[self.id] = time.time() - self.t0
        else:
            self.g_times[self.id] = time.time() - self.t0 + 0.5

    def send_global_verdict_to_cluster_members(self, s_id, gv):
        latency = self.latency_matrix[self.id][s_id]
        if latency >= 0:
            time.sleep(latency)
        self.gv_queues[s_id].put(gv)
