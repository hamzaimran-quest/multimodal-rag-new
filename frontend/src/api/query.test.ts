import { describe, expect, it } from "vitest";

import { parseSseChunk } from "./query";

describe("query SSE parser", () => {
  it("parses token and sources events", () => {
    const chunk =
      'event: token\ndata: {"token":"Revenue "}\n\n' +
      'event: token\ndata: {"token":"grew"}\n\n' +
      'event: sources\ndata: {"sources":[{"chunk_id":"c1","filename":"huawei.pdf","page_number":12,"chunk_type":"text","snippet":"abc","score":0.9}]}\n\n';

    const events = parseSseChunk(chunk);
    expect(events).toHaveLength(3);
    expect(events[0]).toEqual({ event: "token", data: { token: "Revenue " } });
    expect(events[2].event).toBe("sources");
  });

  it("parses sql and route events", () => {
    const chunk =
      'event: route\ndata: {"mode":"hybrid"}\n\n' +
      'event: sql\ndata: {"connection_id":1,"display_name":"Analytics","route_mode":"hybrid"}\n\n';

    const events = parseSseChunk(chunk);
    expect(events).toHaveLength(2);
    expect(events[0]).toEqual({ event: "route", data: { mode: "hybrid" } });
    expect(events[1].event).toBe("sql");
  });
  it("ignores malformed frames", () => {
    const chunk = "event: token\ndata: {bad}\n\n";
    const events = parseSseChunk(chunk);
    expect(events).toEqual([]);
  });
});

