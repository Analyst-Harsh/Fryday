import { describe, expect, it } from "vitest";
import { hello, parse, played } from "./protocol";

describe("parse", () => {
  it("accepts our version and known types", () => {
    expect(parse('{"v":1,"t":"speech","turn":3}')).toEqual({ v: 1, t: "speech", turn: 3 });
  });

  it("rejects bad JSON, a wrong or missing version, and unknown types", () => {
    for (const text of ["not json", "[]", "null", '{"t":"speech"}', '{"v":2,"t":"speech"}', '{"v":1,"t":"party"}']) {
      expect(parse(text)).toBeNull();
    }
  });
});

describe("client messages", () => {
  it("carry the protocol version", () => {
    expect(JSON.parse(hello("abc"))).toEqual({ v: 1, t: "hello", token: "abc" });
    expect(JSON.parse(played(4, 1200))).toEqual({ v: 1, t: "played", turn: 4, bytes: 1200 });
  });
});
