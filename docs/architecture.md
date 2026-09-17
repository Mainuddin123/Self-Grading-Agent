\# Self-Grading RAG Agent — Architecture



User Question

&#x20;     |

&#x20;     v

Question Processing

&#x20;     |

&#x20;     v

Document Retrieval

&#x20;     |

&#x20;     v

Relevant Evidence

&#x20;     |

&#x20;     +----------------------+

&#x20;     |                      |

&#x20;     v                      v

Evidence Support       No Relevant Evidence

Check                       |

&#x20;     |                      v

&#x20;     v               NO\_SUPPORTED\_ANSWER

Answer Generation

&#x20;     |

&#x20;     v

Self-Grading

&#x20;     |

&#x20;     +----------------+----------------+ 

&#x20;     |                |                |

&#x20;     v                v                v

Correctness       Relevance       Grounding

&#x20;     |                |                |

&#x20;     +----------------+----------------+

&#x20;                      |

&#x20;                      v

&#x20;                Overall Score

&#x20;                      |

&#x20;                      v

&#x20;                  Confidence

&#x20;                      |

&#x20;                      v

&#x20;                  Final Result

