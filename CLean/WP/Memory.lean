import CLean.WP.Expr

namespace CLean
namespace WP

def ByteRead (mem : ByteMem) (offset : Nat) (bytes : List Byte) : Prop :=
  Helpers.readBytes? mem offset bytes.length = some bytes

def ByteWrite (mem mem' : ByteMem) (offset : Nat) (bytes : List Byte) : Prop :=
  mem' = Helpers.writeBytes mem offset bytes

def EncodedScalar (ty : ScalarTy) (value : Value) (bytes : List Byte) : Prop :=
  Helpers.encodeScalar? ty value = some bytes

def DecodedScalar (ty : ScalarTy) (bytes : List Byte) (value : Value) : Prop :=
  Helpers.decodeScalar? ty bytes = some value

def BaseMemory (st : State) (addr : Addr) (mem : ByteMem) : Prop :=
  Helpers.getSpaceBaseMem? st addr = some mem

def SetBaseMemory (st : State) (addr : Addr) (mem : ByteMem) (st' : State) : Prop :=
  Helpers.setSpaceBaseMem? st addr mem = some st'

def ReadMemFact (st : State) (space : AddrSpace) (ty : ScalarTy) (addr : Addr)
    (value : Value) : Prop :=
  Helpers.readMem? st space ty addr = some value

def WriteMemFact (st : State) (space : AddrSpace) (ty : ScalarTy) (addr : Addr)
    (value : Value) (st' : State) : Prop :=
  Helpers.writeMem? st space ty addr value = some st'

theorem byteRead_of_eq {mem : ByteMem} {offset : Nat} {bytes : List Byte}
    (hread : Helpers.readBytes? mem offset bytes.length = some bytes) :
    ByteRead mem offset bytes :=
  hread

theorem byteWrite_of_eq {mem mem' : ByteMem} {offset : Nat} {bytes : List Byte}
    (hwrite : mem' = Helpers.writeBytes mem offset bytes) :
    ByteWrite mem mem' offset bytes :=
  hwrite

theorem encodedScalar_of_eq {ty : ScalarTy} {value : Value} {bytes : List Byte}
    (hencode : Helpers.encodeScalar? ty value = some bytes) :
    EncodedScalar ty value bytes :=
  hencode

theorem decodedScalar_of_eq {ty : ScalarTy} {bytes : List Byte} {value : Value}
    (hdecode : Helpers.decodeScalar? ty bytes = some value) :
    DecodedScalar ty bytes value :=
  hdecode

private theorem writeBytes_loop_preserves_before
    {offset i j : Nat} {acc : ByteMem} {bytes : List Byte}
    (hlt : j < i) :
    (Helpers.writeBytes.loop offset i acc bytes)[offset + j]? = acc[offset + j]? := by
  induction bytes generalizing i acc with
  | nil =>
      rw [Helpers.writeBytes.loop.eq_def]
  | cons byte bytes ih =>
      rw [Helpers.writeBytes.loop.eq_def]
      rw [ih (i := i + 1) (acc := Std.HashMap.insert acc (offset + i) byte) (by omega)]
      rw [Std.HashMap.getElem?_insert]
      have hneq : ((offset + i == offset + j) = false) := by
        apply Bool.eq_false_iff.mpr
        intro hEq
        have hnat : offset + i = offset + j := by
          exact beq_iff_eq.mp hEq
        omega
      simp [hneq]

private theorem memoryBytes_writeBytes_loop
    {mem : ByteMem} {offset i : Nat} {bytes : List Byte} :
    CSL.memoryBytes (Helpers.writeBytes.loop offset i mem bytes) (offset + i) bytes := by
  induction bytes generalizing i mem with
  | nil =>
      simp [CSL.memoryBytes]
  | cons byte bytes ih =>
      rw [Helpers.writeBytes.loop.eq_def]
      constructor
      · unfold CSL.memoryByte
        rw [writeBytes_loop_preserves_before (bytes := bytes) (j := i) (i := i + 1) (by omega)]
        simp
      · simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using
          (ih (i := i + 1) (mem := Std.HashMap.insert mem (offset + i) byte))

theorem memoryBytes_writeBytes
    {mem : ByteMem} {offset : Nat} {bytes : List Byte} :
    CSL.memoryBytes (Helpers.writeBytes mem offset bytes) offset bytes := by
  unfold Helpers.writeBytes
  simpa using
    (memoryBytes_writeBytes_loop (mem := mem) (offset := offset) (i := 0) (bytes := bytes))

private theorem writeBytes_loop_preserves_not_in_range
    {mem : ByteMem} {base i key : Nat} {bytes : List Byte}
    (hnot : ∀ k, i ≤ k → k < i + bytes.length → key ≠ base + k) :
    (Helpers.writeBytes.loop base i mem bytes)[key]? = mem[key]? := by
  induction bytes generalizing i mem with
  | nil =>
      rw [Helpers.writeBytes.loop.eq_def]
  | cons byte bytes ih =>
      rw [Helpers.writeBytes.loop.eq_def]
      rw [ih (i := i + 1) (mem := Std.HashMap.insert mem (base + i) byte) (by
        intro k hik hk
        exact hnot k (by omega)
          (by simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using hk))]
      rw [Std.HashMap.getElem?_insert]
      have hneq : ((base + i == key) = false) := by
        apply Bool.eq_false_iff.mpr
        intro heq
        have hkey : base + i = key := beq_iff_eq.mp heq
        exact hnot i (by omega) (by simp) hkey.symm
      simp [hneq]

theorem writeBytes_preserves_of_not_in_range
    {mem : ByteMem} {base key : Nat} {bytes : List Byte}
    (hnot : key < base ∨ base + bytes.length ≤ key) :
    (Helpers.writeBytes mem base bytes)[key]? = mem[key]? := by
  unfold Helpers.writeBytes
  exact writeBytes_loop_preserves_not_in_range (base := base) (i := 0) (key := key)
    (bytes := bytes) (by
      intro k _hik hk hkey
      rcases hnot with hlt | hle
      · omega
      · omega)

def ByteRangesDisjoint (offset₁ len₁ offset₂ len₂ : Nat) : Prop :=
  offset₁ + len₁ ≤ offset₂ ∨ offset₂ + len₂ ≤ offset₁

theorem ByteRangesDisjoint.symm {offset₁ len₁ offset₂ len₂ : Nat}
    (h : ByteRangesDisjoint offset₁ len₁ offset₂ len₂) :
    ByteRangesDisjoint offset₂ len₂ offset₁ len₁ := by
  unfold ByteRangesDisjoint at h ⊢
  rcases h with hbefore | hafter
  · exact Or.inr hbefore
  · exact Or.inl hafter

theorem memoryBytes_writeBytes_preserved_of_disjoint
    {mem : ByteMem} {readOffset writeOffset : Nat} {oldBytes newBytes : List Byte}
    (hdisjoint :
      ByteRangesDisjoint readOffset oldBytes.length writeOffset newBytes.length)
    (hmem : CSL.memoryBytes mem readOffset oldBytes) :
    CSL.memoryBytes (Helpers.writeBytes mem writeOffset newBytes) readOffset oldBytes := by
  induction oldBytes generalizing readOffset with
  | nil =>
      simp [CSL.memoryBytes]
  | cons byte bytes ih =>
      rcases hmem with ⟨hbyte, hbytes⟩
      constructor
      · unfold CSL.memoryByte at hbyte ⊢
        rw [writeBytes_preserves_of_not_in_range]
        · exact hbyte
        · unfold ByteRangesDisjoint at hdisjoint
          rcases hdisjoint with hbefore | hafter
          · exact Or.inl (Nat.lt_of_lt_of_le (by simp) hbefore)
          · exact Or.inr hafter
      · apply ih
        · unfold ByteRangesDisjoint at hdisjoint ⊢
          rcases hdisjoint with hbefore | hafter
          · exact Or.inl (by
              simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using hbefore)
          · exact Or.inr (Nat.le_trans hafter (Nat.le_succ readOffset))
        · exact hbytes

theorem byteWrite_memoryBytes
    {mem mem' : ByteMem} {offset : Nat} {bytes : List Byte}
    (hwrite : ByteWrite mem mem' offset bytes) :
    CSL.memoryBytes mem' offset bytes := by
  subst mem'
  exact memoryBytes_writeBytes

private theorem readBytes_loop_of_memoryBytes
    {mem : ByteMem} {base i : Nat} {acc bytes : List Byte}
    (hmem : CSL.memoryBytes mem (base + i) bytes) :
    Helpers.readBytes?.loop mem base (i + bytes.length) i acc =
      some (acc.reverse ++ bytes) := by
  induction bytes generalizing i acc with
  | nil =>
      rw [Helpers.readBytes?.loop.eq_def]
      simp
  | cons byte bytes ih =>
      simp [CSL.memoryBytes, CSL.memoryByte] at hmem
      rw [Helpers.readBytes?.loop.eq_def]
      have hlt : i < i + (byte :: bytes).length := by simp
      simp [hlt, hmem.1]
      have htail : CSL.memoryBytes mem (base + (i + 1)) bytes := by
        simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm] using hmem.2
      simpa [Nat.add_assoc, Nat.add_comm, Nat.add_left_comm, List.append_assoc] using
        (ih (i := i + 1) (acc := byte :: acc) htail)

theorem byteRead_of_memoryBytes
    {mem : ByteMem} {offset : Nat} {bytes : List Byte}
    (hmem : CSL.memoryBytes mem offset bytes) :
    ByteRead mem offset bytes := by
  unfold ByteRead Helpers.readBytes?
  simpa using
    (readBytes_loop_of_memoryBytes (mem := mem) (base := offset) (i := 0) (acc := []) hmem)

theorem byteRead_of_globalBytes
    {st : State} {r : CSL.Resource} {offset : Nat} {perm : CSL.BytePerm}
    {bytes : List Byte}
    (hbytes : CSL.globalBytes offset perm bytes st r) :
    ByteRead st.global.bytes offset bytes :=
  byteRead_of_memoryBytes (CSL.globalBytes_memory hbytes)

theorem byteRead_of_sharedBytes
    {st : State} {r : CSL.Resource} {cta : CTAId} {offset : Nat}
    {perm : CSL.BytePerm} {bytes : List Byte} {ctaState : CTAState}
    (hbytes : CSL.sharedBytes cta offset perm bytes st r)
    (hcta : st.getCTA? cta = some ctaState) :
    ByteRead ctaState.shared.bytes offset bytes :=
  byteRead_of_memoryBytes (CSL.sharedBytes_memory hbytes ctaState hcta)

theorem byteRead_of_localBytes
    {st : State} {r : CSL.Resource} {cta : CTAId} {warp : WarpId} {lane : LaneId}
    {offset : Nat} {perm : CSL.BytePerm} {bytes : List Byte} {laneState : LaneState}
    (hbytes : CSL.localBytes cta warp lane offset perm bytes st r)
    (hlane : st.getLane? cta warp lane = some laneState) :
    ByteRead laneState.localMem.bytes offset bytes :=
  byteRead_of_memoryBytes (CSL.localBytes_memory hbytes laneState hlane)

theorem byteRead_of_paramBytes
    {st : State} {r : CSL.Resource} {offset : Nat} {bytes : List Byte}
    (hbytes : CSL.paramBytes offset bytes st r) :
    ByteRead st.param.bytes offset bytes :=
  byteRead_of_memoryBytes (CSL.paramBytes_memory hbytes)

theorem byteRead_of_constBytes
    {st : State} {r : CSL.Resource} {offset : Nat} {bytes : List Byte}
    (hbytes : CSL.constBytes offset bytes st r) :
    ByteRead st.const.bytes offset bytes :=
  byteRead_of_memoryBytes (CSL.constBytes_memory hbytes)

theorem readMem_of_byteRead
    {st : State} {space : AddrSpace} {ty : ScalarTy} {addr : Addr}
    {mem : ByteMem} {bytes : List Byte} {value : Value}
    (haccess : AccessOk space ty addr)
    (hbase : BaseMemory st addr mem)
    (hwidth : Typing.byteWidth? ty = some bytes.length)
    (hread : ByteRead mem addr.offset bytes)
    (hdecode : DecodedScalar ty bytes value) :
    ReadMemFact st space ty addr value := by
  unfold ReadMemFact Helpers.readMem?
  simp [AccessOk] at haccess
  simp [BaseMemory] at hbase
  simp [ByteRead] at hread
  simp [DecodedScalar] at hdecode
  simp [haccess, hbase, hwidth]
  rw [hread]
  exact hdecode

theorem writeMem_of_byteWrite
    {st st' : State} {space : AddrSpace} {ty : ScalarTy} {addr : Addr}
    {mem mem' : ByteMem} {bytes : List Byte} {value : Value}
    (haccess : AccessOk space ty addr)
    (hencode : EncodedScalar ty value bytes)
    (hbase : BaseMemory st addr mem)
    (hwrite : ByteWrite mem mem' addr.offset bytes)
    (hset : SetBaseMemory st addr mem' st') :
    WriteMemFact st space ty addr value st' := by
  unfold WriteMemFact Helpers.writeMem?
  subst mem'
  simp [AccessOk] at haccess
  simp [BaseMemory] at hbase
  simp [SetBaseMemory] at hset
  simp [EncodedScalar] at hencode
  simp [haccess, hbase, hencode, hset]

theorem globalReadMem_of_byteRead
    {st : State} {ty : ScalarTy} {offset : Nat} {bytes : List Byte} {value : Value}
    (haccess : AccessOk .global ty (.global offset))
    (hwidth : Typing.byteWidth? ty = some bytes.length)
    (hread : ByteRead st.global.bytes offset bytes)
    (hdecode : DecodedScalar ty bytes value) :
    ReadMemFact st .global ty (.global offset) value :=
  readMem_of_byteRead (mem := st.global.bytes) haccess
    (by simp [BaseMemory, Helpers.getSpaceBaseMem?])
    hwidth (by simpa [Addr.offset] using hread) hdecode

theorem paramReadMem_of_byteRead
    {st : State} {ty : ScalarTy} {offset : Nat} {bytes : List Byte} {value : Value}
    (haccess : AccessOk .param ty (.param offset))
    (hwidth : Typing.byteWidth? ty = some bytes.length)
    (hread : ByteRead st.param.bytes offset bytes)
    (hdecode : DecodedScalar ty bytes value) :
    ReadMemFact st .param ty (.param offset) value :=
  readMem_of_byteRead (mem := st.param.bytes) haccess
    (by simp [BaseMemory, Helpers.getSpaceBaseMem?])
    hwidth (by simpa [Addr.offset] using hread) hdecode

theorem constReadMem_of_byteRead
    {st : State} {ty : ScalarTy} {offset : Nat} {bytes : List Byte} {value : Value}
    (haccess : AccessOk .const ty (.const offset))
    (hwidth : Typing.byteWidth? ty = some bytes.length)
    (hread : ByteRead st.const.bytes offset bytes)
    (hdecode : DecodedScalar ty bytes value) :
    ReadMemFact st .const ty (.const offset) value :=
  readMem_of_byteRead (mem := st.const.bytes) haccess
    (by simp [BaseMemory, Helpers.getSpaceBaseMem?])
    hwidth (by simpa [Addr.offset] using hread) hdecode

theorem sharedReadMem_of_byteRead
    {st : State} {cta : CTAId} {ty : ScalarTy} {offset : Nat}
    {ctaState : CTAState} {bytes : List Byte} {value : Value}
    (haccess : AccessOk .shared ty (.shared cta offset))
    (hcta : st.getCTA? cta = some ctaState)
    (hwidth : Typing.byteWidth? ty = some bytes.length)
    (hread : ByteRead ctaState.shared.bytes offset bytes)
    (hdecode : DecodedScalar ty bytes value) :
    ReadMemFact st .shared ty (.shared cta offset) value :=
  readMem_of_byteRead (mem := ctaState.shared.bytes) haccess
    (by simp [BaseMemory, Helpers.getSpaceBaseMem?, hcta])
    hwidth (by simpa [Addr.offset] using hread) hdecode

theorem localReadMem_of_byteRead
    {st : State} {cta : CTAId} {warp : WarpId} {lane : LaneId}
    {ty : ScalarTy} {offset : Nat} {laneState : LaneState}
    {bytes : List Byte} {value : Value}
    (haccess : AccessOk .local ty (.local cta warp lane offset))
    (hlane : st.getLane? cta warp lane = some laneState)
    (hwidth : Typing.byteWidth? ty = some bytes.length)
    (hread : ByteRead laneState.localMem.bytes offset bytes)
    (hdecode : DecodedScalar ty bytes value) :
    ReadMemFact st .local ty (.local cta warp lane offset) value :=
  readMem_of_byteRead (mem := laneState.localMem.bytes) haccess
    (by simp [BaseMemory, Helpers.getSpaceBaseMem?, hlane])
    hwidth (by simpa [Addr.offset] using hread) hdecode

theorem globalReadMem_of_globalBytes
    {st : State} {r : CSL.Resource} {ty : ScalarTy} {offset : Nat}
    {perm : CSL.BytePerm} {bytes : List Byte} {value : Value}
    (haccess : AccessOk .global ty (.global offset))
    (hwidth : Typing.byteWidth? ty = some bytes.length)
    (hbytes : CSL.globalBytes offset perm bytes st r)
    (hdecode : DecodedScalar ty bytes value) :
    ReadMemFact st .global ty (.global offset) value :=
  globalReadMem_of_byteRead haccess hwidth (byteRead_of_globalBytes hbytes) hdecode

theorem paramReadMem_of_paramBytes
    {st : State} {r : CSL.Resource} {ty : ScalarTy} {offset : Nat}
    {bytes : List Byte} {value : Value}
    (haccess : AccessOk .param ty (.param offset))
    (hwidth : Typing.byteWidth? ty = some bytes.length)
    (hbytes : CSL.paramBytes offset bytes st r)
    (hdecode : DecodedScalar ty bytes value) :
    ReadMemFact st .param ty (.param offset) value :=
  paramReadMem_of_byteRead haccess hwidth (byteRead_of_paramBytes hbytes) hdecode

theorem constReadMem_of_constBytes
    {st : State} {r : CSL.Resource} {ty : ScalarTy} {offset : Nat}
    {bytes : List Byte} {value : Value}
    (haccess : AccessOk .const ty (.const offset))
    (hwidth : Typing.byteWidth? ty = some bytes.length)
    (hbytes : CSL.constBytes offset bytes st r)
    (hdecode : DecodedScalar ty bytes value) :
    ReadMemFact st .const ty (.const offset) value :=
  constReadMem_of_byteRead haccess hwidth (byteRead_of_constBytes hbytes) hdecode

theorem sharedReadMem_of_sharedBytes
    {st : State} {r : CSL.Resource} {cta : CTAId} {ty : ScalarTy} {offset : Nat}
    {perm : CSL.BytePerm} {ctaState : CTAState} {bytes : List Byte} {value : Value}
    (haccess : AccessOk .shared ty (.shared cta offset))
    (hcta : st.getCTA? cta = some ctaState)
    (hwidth : Typing.byteWidth? ty = some bytes.length)
    (hbytes : CSL.sharedBytes cta offset perm bytes st r)
    (hdecode : DecodedScalar ty bytes value) :
    ReadMemFact st .shared ty (.shared cta offset) value :=
  sharedReadMem_of_byteRead haccess hcta hwidth
    (byteRead_of_sharedBytes hbytes hcta) hdecode

theorem localReadMem_of_localBytes
    {st : State} {r : CSL.Resource} {cta : CTAId} {warp : WarpId} {lane : LaneId}
    {ty : ScalarTy} {offset : Nat} {perm : CSL.BytePerm}
    {laneState : LaneState} {bytes : List Byte} {value : Value}
    (haccess : AccessOk .local ty (.local cta warp lane offset))
    (hlane : st.getLane? cta warp lane = some laneState)
    (hwidth : Typing.byteWidth? ty = some bytes.length)
    (hbytes : CSL.localBytes cta warp lane offset perm bytes st r)
    (hdecode : DecodedScalar ty bytes value) :
    ReadMemFact st .local ty (.local cta warp lane offset) value :=
  localReadMem_of_byteRead haccess hlane hwidth
    (byteRead_of_localBytes hbytes hlane) hdecode

theorem globalWriteMem_of_byteWrite
    {st : State} {ty : ScalarTy} {offset : Nat} {bytes : List Byte} {value : Value}
    {mem' : ByteMem}
    (haccess : AccessOk .global ty (.global offset))
    (hencode : EncodedScalar ty value bytes)
    (hwrite : ByteWrite st.global.bytes mem' offset bytes) :
    WriteMemFact st .global ty (.global offset) value { st with global := { bytes := mem' } } :=
  writeMem_of_byteWrite (mem := st.global.bytes) (mem' := mem') haccess hencode
    (by simp [BaseMemory, Helpers.getSpaceBaseMem?])
    (by simpa [Addr.offset] using hwrite)
    (by simp [SetBaseMemory, Helpers.setSpaceBaseMem?])

theorem sharedWriteMem_of_byteWrite
    {st : State} {cta : CTAId} {ty : ScalarTy} {offset : Nat}
    {ctaState : CTAState} {bytes : List Byte} {value : Value} {mem' : ByteMem}
    (haccess : AccessOk .shared ty (.shared cta offset))
    (hencode : EncodedScalar ty value bytes)
    (hcta : st.getCTA? cta = some ctaState)
    (hwrite : ByteWrite ctaState.shared.bytes mem' offset bytes) :
    WriteMemFact st .shared ty (.shared cta offset) value
      (st.setCTA cta { ctaState with shared := { bytes := mem' } }) :=
  writeMem_of_byteWrite (mem := ctaState.shared.bytes) (mem' := mem') haccess hencode
    (by simp [BaseMemory, Helpers.getSpaceBaseMem?, hcta])
    (by simpa [Addr.offset] using hwrite)
    (by simp [SetBaseMemory, Helpers.setSpaceBaseMem?, hcta])

theorem localWriteMem_of_byteWrite
    {st st' : State} {cta : CTAId} {warp : WarpId} {lane : LaneId}
    {ty : ScalarTy} {offset : Nat} {laneState : LaneState}
    {bytes : List Byte} {value : Value} {mem' : ByteMem}
    (haccess : AccessOk .local ty (.local cta warp lane offset))
    (hencode : EncodedScalar ty value bytes)
    (hlane : st.getLane? cta warp lane = some laneState)
    (hwrite : ByteWrite laneState.localMem.bytes mem' offset bytes)
    (hset :
      st.setLane cta warp lane { laneState with localMem := { bytes := mem' } } = some st') :
    WriteMemFact st .local ty (.local cta warp lane offset) value st' :=
  writeMem_of_byteWrite (mem := laneState.localMem.bytes) (mem' := mem') haccess hencode
    (by simp [BaseMemory, Helpers.getSpaceBaseMem?, hlane])
    (by simpa [Addr.offset] using hwrite)
    (by simp [SetBaseMemory, Helpers.setSpaceBaseMem?, hlane, hset])

end WP
end CLean
